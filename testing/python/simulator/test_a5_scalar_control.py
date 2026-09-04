# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Device-free differential tests for the bounded A5 scalar-control bridge."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tilelang.simulator import UnsupportedSimOpError, default_timing_profile
from tilelang.simulator.bridge import _TirBridge


@dataclass(frozen=True)
class _Imm:
    value: int | float | bool
    dtype: str = "int32"


class _Var:

    def __init__(self, name: str, dtype: str = "int32") -> None:
        self.name = name
        self.dtype = dtype


@dataclass(frozen=True)
class _Binary:
    a: object
    b: object


class _Add(_Binary):
    pass


class _Mul(_Binary):
    pass


class _Sub(_Binary):
    pass


class _EQ(_Binary):
    pass


class _LT(_Binary):
    pass


class _GT(_Binary):
    pass


class _GE(_Binary):
    pass


class _And(_Binary):
    pass


@dataclass(frozen=True)
class _Select:
    condition: object
    true_value: object
    false_value: object


@dataclass(frozen=True)
class _Buffer:
    name: str
    data: object
    shape: tuple[object, ...]
    dtype: str


@dataclass(frozen=True)
class _BufferLoad:
    buffer: _Buffer
    indices: tuple[object, ...]


@dataclass(frozen=True)
class _BufferStore:
    buffer: _Buffer
    value: object
    indices: tuple[object, ...]


@dataclass(frozen=True)
class _SeqStmt:
    seq: tuple[object, ...]


@dataclass(frozen=True)
class _For:
    loop_var: _Var
    min: object
    extent: object
    body: object


@dataclass(frozen=True)
class _IfThenElse:
    condition: object
    then_case: object
    else_case: object | None = None


@dataclass(frozen=True)
class _LetStmt:
    var: _Var
    value: object
    body: object


@dataclass(frozen=True)
class _Call:
    op: object
    args: tuple[object, ...] = ()


@dataclass(frozen=True)
class _PrimFunc:
    body: object
    attrs: dict[str, object] | None = None
    buffer_map: dict[object, _Buffer] | None = None

    def __post_init__(self) -> None:
        if self.buffer_map is None:
            object.__setattr__(self, "buffer_map", {})


class _UnusedStmt:
    pass


class _Analysis:

    @staticmethod
    def undefined_vars(_value: object, _bound: list[object]) -> list[object]:
        return []


class _StmtFunctor:

    @staticmethod
    def substitute(value: object, _replacements: dict[object, object]) -> object:
        return value


class _Analyzer:

    @staticmethod
    def simplify(value: object) -> object:
        return value


class _FakeTir:
    PrimFunc = _PrimFunc
    SeqStmt = _SeqStmt
    AttrStmt = _UnusedStmt
    For = _For
    IfThenElse = _IfThenElse
    LetStmt = _LetStmt
    Allocate = _UnusedStmt
    DeclBuffer = _UnusedStmt
    BufferRealize = _UnusedStmt
    BlockRealize = _UnusedStmt
    Block = _UnusedStmt
    AssertStmt = _UnusedStmt
    Evaluate = _UnusedStmt
    Call = _Call
    BufferStore = _BufferStore
    BufferLoad = _BufferLoad
    Var = _Var
    IntImm = _Imm
    FloatImm = _Imm
    Add = _Add
    Mul = _Mul
    Sub = _Sub
    EQ = _EQ
    LT = _LT
    GT = _GT
    GE = _GE
    And = _And
    Select = _Select
    analysis = _Analysis()
    stmt_functor = _StmtFunctor()

    @staticmethod
    def const(value: int | float | bool, dtype: str | None = None) -> _Imm:
        return _Imm(value, dtype or "int32")


def _make_bridge(
    platform: str, symbol_bindings: dict[str, int | float | bool] | None = None
) -> _TirBridge:
    return _TirBridge(
        tvm=SimpleNamespace(IRModule=type("IRModule", (), {})),
        tir=_FakeTir,
        analyzer=_Analyzer(),
        platform=platform,
        timing_profile=default_timing_profile(platform),
        max_unrolled_iterations=32,
        symbol_bindings=symbol_bindings or {},
    )


def _mask_program() -> _PrimFunc:
    index = _Var("i")
    temporary = _Var("tmp")
    mask = _Buffer("mask", _Var("mask_data", "handle"), (_Imm(1),), "int32")
    output = _Buffer("output", _Var("output_data", "handle"), (_Imm(4),), "int32")
    body = _SeqStmt((
        _BufferStore(
            mask,
            _Select(_LT(index, _Imm(2)), _Imm(0), _Imm(1)),
            (_Imm(0),),
        ),
        _IfThenElse(
            _EQ(_BufferLoad(mask, (_Imm(0),)), _Imm(0)),
            _LetStmt(
                temporary,
                _Add(index, _Imm(1)),
                _BufferStore(
                    output,
                    _Select(
                        _EQ(temporary, _Imm(2)),
                        _Add(_Mul(temporary, _Imm(2)), _Imm(1)),
                        _Mul(temporary, _Imm(2)),
                    ),
                    (index,),
                ),
            ),
        ),
    ))
    return _PrimFunc(_For(index, _Imm(0), _Imm(4), body))


def test_a5_scalar_control_matches_small_python_truth() -> None:
    program = _make_bridge("A5").build(_mask_program())

    observed = {
        task.metadata["scalar_indices"][0]: task.metadata["scalar_value"]
        for task in program.tasks
        if task.metadata["buffer"] == "output"
    }
    expected = {
        index: (index + 1) * 2 + (1 if index + 1 == 2 else 0)
        for index in range(4)
        if int(index < 2) == 1
    }

    assert observed == expected == {0: 2, 1: 5}


def test_a5_matches_e288_case0_mask_control_fragment() -> None:
    sk_index = _Var("sk_i")
    relative = _Var("relative")
    causal = _Var("causal")
    window_left = _Var("window_left")
    window_right = _Var("window_right")
    mask = _Buffer("mask", _Var("mask_data", "handle"), (_Imm(32),), "int32")
    output = _Buffer("output", _Var("output_data", "handle"), (_Imm(5),), "int32")
    mask_body = _SeqStmt(
        (
            _BufferStore(mask, _Imm(0), (_Imm(0),)),
            _IfThenElse(
                _And(_EQ(causal, _Imm(1)), _GT(relative, _Imm(0))),
                _BufferStore(mask, _Imm(1), (_Imm(0),)),
            ),
            _IfThenElse(
                _And(
                    _GE(window_left, _Imm(0)),
                    _LT(relative, _Sub(_Imm(0), window_left)),
                ),
                _BufferStore(mask, _Imm(1), (_Imm(0),)),
            ),
            _IfThenElse(
                _And(_GE(window_right, _Imm(0)), _GT(relative, window_right)),
                _BufferStore(mask, _Imm(1), (_Imm(0),)),
            ),
            _IfThenElse(
                _EQ(_BufferLoad(mask, (_Imm(0),)), _Imm(0)),
                _BufferStore(output, relative, (sk_index,)),
            ),
        )
    )
    function = _PrimFunc(
        _For(
            sk_index,
            _Imm(0),
            _Imm(5),
            _LetStmt(relative, _Sub(sk_index, _Imm(2)), mask_body),
        )
    )

    program = _make_bridge(
        "A5",
        {"causal": True, "window_left": -1, "window_right": 0},
    ).build(function)
    observed = {
        task.metadata["scalar_indices"][0]: task.metadata["scalar_value"]
        for task in program.tasks
        if task.metadata["buffer"] == "output"
    }
    expected = {
        sk: sk - 2
        for sk in range(5)
        if not (True and sk - 2 > 0)
        and not (-1 >= 0 and sk - 2 < 1)
        and not (0 >= 0 and sk - 2 > 0)
    }

    assert observed == expected == {0: -2, 1: -1, 2: 0}


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_a2_a3_keep_dynamic_buffer_condition_boundary(platform: str) -> None:
    with pytest.raises(
            UnsupportedSimOpError,
            match=r"dynamic if condition.*first A2/A3 simulator bridge",
    ):
        _make_bridge(platform).build(_mask_program())


def test_a5_scalar_buffer_read_before_write_fails_closed() -> None:
    mask = _Buffer("mask", _Var("mask_data", "handle"), (_Imm(1),), "int32")
    function = _PrimFunc(_IfThenElse(_EQ(_BufferLoad(mask, (_Imm(0),)), _Imm(0)), None))

    with pytest.raises(
            UnsupportedSimOpError,
            match=r"A5 scalar if condition.*mask\[0\].*read before write",
    ):
        _make_bridge("A5").build(function)


def test_a5_storage_rewrite_aliases_do_not_collide_by_index() -> None:
    shared_data = _Var("shared_ub", "handle")
    scratch = _Buffer("scratch", shared_data, (_Imm(32),), "int32")
    mask = _Buffer("mask", shared_data, (_Imm(32),), "int32")
    output = _Buffer("output", _Var("output_data", "handle"), (_Imm(1),), "int32")
    function = _PrimFunc(
        _SeqStmt(
            (
                _BufferStore(scratch, _Imm(7), (_Imm(0),)),
                _BufferStore(mask, _Imm(0), (_Imm(0),)),
                _IfThenElse(
                    _EQ(_BufferLoad(mask, (_Imm(0),)), _Imm(0)),
                    _BufferStore(output, _Imm(1), (_Imm(0),)),
                ),
            )
        )
    )

    program = _make_bridge("A5").build(function)

    assert [
        task.metadata["scalar_value"]
        for task in program.tasks
        if task.metadata["buffer"] == "output"
    ] == [1]


def test_a5_unknown_scalar_expression_fails_closed() -> None:
    unknown = _Call(SimpleNamespace(name="tir.future_scalar"))
    function = _PrimFunc(_IfThenElse(unknown, None))

    with pytest.raises(
            UnsupportedSimOpError,
            match=r"A5 scalar if condition.*unsupported scalar expression _Call",
    ):
        _make_bridge("A5").build(function)


def test_a5_unknown_scalar_store_fails_when_control_consumes_it() -> None:
    mask = _Buffer("mask", _Var("mask_data", "handle"), (_Imm(1),), "int32")
    unknown = _Call(SimpleNamespace(name="tir.future_scalar"))
    function = _PrimFunc(
        _SeqStmt(
            (
                _BufferStore(mask, unknown, (_Imm(0),)),
                _IfThenElse(_EQ(_BufferLoad(mask, (_Imm(0),)), _Imm(0)), None),
            )
        )
    )

    with pytest.raises(
        UnsupportedSimOpError,
        match=r"A5 scalar if condition.*unsupported scalar expression _Call",
    ):
        _make_bridge("A5").build(function)
