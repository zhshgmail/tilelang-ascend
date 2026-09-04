# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Bounded scalar-control semantics for the A5 final-TIR bridge."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
import re
from typing import Any

from .errors import UnsupportedSimOpError
from .program import MemoryScope


ScalarValue = bool | int | float
_BufferToken = tuple[Any, str]
_Cell = tuple[int | None, int | None, _BufferToken, tuple[int, ...]]


@dataclass(frozen=True)
class _UnknownScalar:
    reason: str


class A5ScalarControlEvaluator:
    """Interpret only scalar values that decide A5 final-TIR control flow."""

    def __init__(
        self,
        *,
        tir: Any,
        analyzer: Any,
        symbol_bindings: Mapping[str, ScalarValue],
    ) -> None:
        self.tir = tir
        self.analyzer = analyzer
        self.symbol_bindings = symbol_bindings
        self.buffer_scope_by_var: dict[Any, MemoryScope] = {}
        self.values: dict[_Cell, ScalarValue | _UnknownScalar] = {}

    def register_buffer(self, buffer_var: Any, scope: MemoryScope) -> None:
        self.buffer_scope_by_var[buffer_var] = scope

    def clear_allocation(
        self,
        buffer_var: Any,
        *,
        core_id: int,
        vector_index: int | None,
    ) -> None:
        scope = self.buffer_scope_by_var.get(buffer_var, MemoryScope.LOCAL)
        owner = (
            (None, None)
            if scope in {MemoryScope.GM, MemoryScope.WORKSPACE}
            else (core_id, vector_index)
        )
        stale = [
            cell
            for cell in self.values
            if cell[:2] == owner and cell[2][0] == buffer_var
        ]
        for cell in stale:
            del self.values[cell]

    def invalidate_call(self, call: Any) -> None:
        """Forget scalar cells touched through a handle by an unmodeled call."""
        handle_vars = {
            var
            for var in self.tir.analysis.undefined_vars(call, [])
            if str(getattr(var, "dtype", "")) == "handle"
        }
        if not handle_vars:
            return
        operation = str(getattr(getattr(call, "op", None), "name", type(call).__name__))
        for cell in tuple(self.values):
            if cell[2][0] in handle_vars:
                self.values[cell] = _UnknownScalar(
                    f"{self._format_token(cell[2], cell[3])} was passed to "
                    f"unmodeled call {operation!r}"
                )

    def require_loop_int(
        self,
        value: Any,
        environment: Mapping[Any, ScalarValue],
        *,
        core_id: int,
        vector_index: int | None,
        what: str,
    ) -> int:
        result = self.require(
            value,
            environment,
            core_id=core_id,
            vector_index=vector_index,
            what=what,
        )
        if isinstance(result, bool) or not isinstance(result, Integral):
            raise UnsupportedSimOpError(
                f"A5 scalar {what} must evaluate to an integer, got {result!r}"
            )
        return int(result)

    def require_condition(
        self,
        value: Any,
        environment: Mapping[Any, ScalarValue],
        *,
        core_id: int,
        vector_index: int | None,
        what: str,
    ) -> bool:
        result = self.require(
            value,
            environment,
            core_id=core_id,
            vector_index=vector_index,
            what=what,
        )
        if not isinstance(result, (bool, Integral)):
            raise UnsupportedSimOpError(
                f"A5 scalar {what} must evaluate to bool or integer, got {result!r}"
            )
        return bool(result)

    def require(
        self,
        value: Any,
        environment: Mapping[Any, ScalarValue],
        *,
        core_id: int,
        vector_index: int | None,
        what: str,
    ) -> ScalarValue:
        result = self._eval(value, environment, core_id, vector_index)
        if isinstance(result, _UnknownScalar):
            raise UnsupportedSimOpError(f"A5 scalar {what} is unsupported: {result.reason}")
        return result

    def record_store(
        self,
        stmt: Any,
        environment: Mapping[Any, ScalarValue],
        *,
        core_id: int,
        vector_index: int | None,
    ) -> dict[str, Any]:
        indices = self._eval_indices(
            stmt.indices, environment, core_id, vector_index
        )
        if isinstance(indices, _UnknownScalar):
            return {}
        value = self._eval(stmt.value, environment, core_id, vector_index)
        if not isinstance(value, _UnknownScalar):
            value = self._cast(value, str(stmt.buffer.dtype))
        cell = self._cell(stmt.buffer, indices, core_id, vector_index)
        self.values[cell] = value
        if isinstance(value, _UnknownScalar):
            return {}
        return {"scalar_indices": indices, "scalar_value": value}

    def _eval(
        self,
        value: Any,
        environment: Mapping[Any, ScalarValue],
        core_id: int,
        vector_index: int | None,
    ) -> ScalarValue | _UnknownScalar:
        literal = self._literal(value)
        if literal is not None:
            return literal
        try:
            if value in environment:
                return environment[value]
        except TypeError:
            pass

        if self._is(value, "Var"):
            name = self._var_name(value)
            if name in self.symbol_bindings:
                return self.symbol_bindings[name]
            return _UnknownScalar(f"unbound scalar Var {name!r}")

        if self._is(value, "BufferLoad"):
            indices = self._eval_indices(
                value.indices, environment, core_id, vector_index
            )
            if isinstance(indices, _UnknownScalar):
                return indices
            cell = self._cell(value.buffer, indices, core_id, vector_index)
            if cell not in self.values:
                return _UnknownScalar(
                    f"{self._format_cell(value.buffer, indices)} read before write"
                )
            return self.values[cell]

        if self._is(value, "Cast"):
            operand = self._eval(value.value, environment, core_id, vector_index)
            if isinstance(operand, _UnknownScalar):
                return operand
            return self._cast(operand, str(value.dtype))

        if self._is(value, "Select"):
            condition = self._eval(
                value.condition, environment, core_id, vector_index
            )
            if isinstance(condition, _UnknownScalar):
                return condition
            if not isinstance(condition, (bool, Integral)):
                return _UnknownScalar(
                    f"Select condition must be bool or integer, got {condition!r}"
                )
            selected = value.true_value if condition else value.false_value
            return self._eval(selected, environment, core_id, vector_index)

        if self._is(value, "Not"):
            operand = self._eval(value.a, environment, core_id, vector_index)
            if isinstance(operand, _UnknownScalar):
                return operand
            if not isinstance(operand, (bool, Integral)):
                return _UnknownScalar(
                    f"Not operand must be bool or integer, got {operand!r}"
                )
            return not bool(operand)

        if self._is(value, "And") or self._is(value, "Or"):
            left = self._eval(value.a, environment, core_id, vector_index)
            if isinstance(left, _UnknownScalar):
                return left
            if not isinstance(left, (bool, Integral)):
                return _UnknownScalar(
                    f"boolean operand must be bool or integer, got {left!r}"
                )
            if self._is(value, "And") and not left:
                return False
            if self._is(value, "Or") and left:
                return True
            right = self._eval(value.b, environment, core_id, vector_index)
            if isinstance(right, _UnknownScalar):
                return right
            if not isinstance(right, (bool, Integral)):
                return _UnknownScalar(
                    f"boolean operand must be bool or integer, got {right!r}"
                )
            return bool(right)

        binary_names = (
            "Add", "Sub", "Mul", "Div", "FloorDiv", "FloorMod", "Mod",
            "TruncDiv", "TruncMod", "Min", "Max", "EQ", "NE", "LT", "LE",
            "GT", "GE",
        )
        operation = next((name for name in binary_names if self._is(value, name)), None)
        if operation is not None:
            left = self._eval(value.a, environment, core_id, vector_index)
            if isinstance(left, _UnknownScalar):
                return left
            right = self._eval(value.b, environment, core_id, vector_index)
            if isinstance(right, _UnknownScalar):
                return right
            return self._apply_binary(operation, left, right)

        simplified = self._simplified(value, environment)
        if simplified is not None:
            return simplified
        return _UnknownScalar(f"unsupported scalar expression {type(value).__name__}")

    @staticmethod
    def _apply_binary(
        operation: str, left: ScalarValue, right: ScalarValue
    ) -> ScalarValue | _UnknownScalar:
        try:
            if operation == "Add":
                result = left + right
            elif operation == "Sub":
                result = left - right
            elif operation == "Mul":
                result = left * right
            elif operation == "Div":
                result = left / right
            elif operation == "FloorDiv":
                result = left // right
            elif operation == "TruncDiv":
                result = int(left / right)
            elif operation == "FloorMod":
                result = left % right
            elif operation in {"Mod", "TruncMod"}:
                quotient = int(left / right)
                result = left - quotient * right
            elif operation == "Min":
                result = min(left, right)
            elif operation == "Max":
                result = max(left, right)
            elif operation == "EQ":
                return left == right
            elif operation == "NE":
                return left != right
            elif operation == "LT":
                return left < right
            elif operation == "LE":
                return left <= right
            elif operation == "GT":
                return left > right
            elif operation == "GE":
                return left >= right
            else:  # pragma: no cover - guarded by the operation tuple above
                return _UnknownScalar(f"unsupported scalar binary operation {operation}")
            if isinstance(result, Real) and not math.isfinite(float(result)):
                return _UnknownScalar(f"scalar {operation} produced non-finite result")
            return result
        except (ArithmeticError, OverflowError, TypeError, ValueError) as error:
            return _UnknownScalar(
                f"scalar {operation} failed for {left!r}, {right!r}: {error}"
            )

    @staticmethod
    def _cast(value: ScalarValue, dtype: str) -> ScalarValue | _UnknownScalar:
        normalized = dtype.strip().lower()
        try:
            if normalized == "bool":
                return bool(value)
            integer = re.fullmatch(r"(u?)int(8|16|32|64)", normalized)
            if integer is not None:
                bits = int(integer.group(2))
                result = int(value) % (1 << bits)
                if not integer.group(1) and result >= (1 << (bits - 1)):
                    result -= 1 << bits
                return result
            if normalized in {"float16", "float32", "float64", "bfloat16"}:
                result = float(value)
                if math.isfinite(result):
                    return result
                return _UnknownScalar(f"cast to {dtype} produced non-finite result")
        except (ArithmeticError, OverflowError, TypeError, ValueError) as error:
            return _UnknownScalar(f"scalar cast to {dtype} failed: {error}")
        return _UnknownScalar(f"unsupported scalar cast dtype {dtype!r}")

    def _eval_indices(
        self,
        values: Any,
        environment: Mapping[Any, ScalarValue],
        core_id: int,
        vector_index: int | None,
    ) -> tuple[int, ...] | _UnknownScalar:
        result = []
        for value in values:
            index = self._eval(value, environment, core_id, vector_index)
            if isinstance(index, _UnknownScalar):
                return _UnknownScalar(f"dynamic scalar buffer index: {index.reason}")
            if isinstance(index, bool) or not isinstance(index, Integral):
                return _UnknownScalar(f"scalar buffer index is not an integer: {index!r}")
            result.append(int(index))
        return tuple(result)

    def _cell(
        self,
        buffer: Any,
        indices: tuple[int, ...],
        core_id: int,
        vector_index: int | None,
    ) -> _Cell:
        data = buffer.data
        scope = self.buffer_scope_by_var.get(data, MemoryScope.LOCAL)
        token = (data, str(buffer.name))
        if scope in {MemoryScope.GM, MemoryScope.WORKSPACE}:
            return None, None, token, indices
        return core_id, vector_index, token, indices

    def _simplified(
        self, value: Any, environment: Mapping[Any, ScalarValue]
    ) -> ScalarValue | None:
        replacements = {
            var: self.tir.const(number, getattr(var, "dtype", None))
            for var, number in environment.items()
        }
        for var in self.tir.analysis.undefined_vars(value, []):
            dtype = str(getattr(var, "dtype", ""))
            name = self._var_name(var)
            if name in self.symbol_bindings and dtype != "handle":
                replacements[var] = self.tir.const(
                    self.symbol_bindings[name], getattr(var, "dtype", None)
                )
        substituted = (
            self.tir.stmt_functor.substitute(value, replacements)
            if replacements
            else value
        )
        return self._literal(self.analyzer.simplify(substituted))

    @staticmethod
    def _literal(value: Any) -> ScalarValue | None:
        literal = value if isinstance(value, (bool, Integral, Real)) else getattr(
            value, "value", None
        )
        if isinstance(literal, bool):
            return literal
        if isinstance(literal, Integral):
            return int(literal)
        if isinstance(literal, Real):
            result = float(literal)
            return result if math.isfinite(result) else None
        return None

    def _is(self, value: Any, name: str) -> bool:
        node_type = getattr(self.tir, name, None)
        return node_type is not None and isinstance(value, node_type)

    @staticmethod
    def _var_name(value: Any) -> str:
        return str(getattr(value, "name", getattr(value, "name_hint", value)))

    @staticmethod
    def _format_cell(buffer: Any, indices: tuple[int, ...]) -> str:
        return f"{buffer.name}[{', '.join(str(index) for index in indices)}]"

    @staticmethod
    def _format_token(token: _BufferToken, indices: tuple[int, ...]) -> str:
        return f"{token[1]}[{', '.join(str(index) for index in indices)}]"
