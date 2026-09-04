# FA-bwd canonical performance adapter

This adapter prepares the exact `e28825ac` one-dispatcher/three-kernel product
for the central `cannbot_perf_probe_v1` measurement.  It does not launch a
device and does not own a performance gate.

The provider must be invoked without changing its locked command:

```bash
python3 "$CONTROLLER" invoke \
  --binding "$PROVIDER_BINDING" \
  --capability performance \
  --op-dir "$PREPARED_OP" \
  --device "$LOGICAL_DEVICE" \
  --receipt-dir "$PROVIDER_MEASUREMENT" \
  --execute
```

The locked command is quick, per-case, warmup 3, active repeats 5, preserves
raw profiles, and measures both Torch reference and candidate on the same
explicit device.  After it exits, run `finalize_perf.py`.  The finalizer rejects
wall-clock data and proportional task splitting.  It extracts only a repeated
eight-run suffix from every raw `task_time` CSV, retains the five active raw
device times, archives all 100 raw profile roots, checks all 50 candidate
processes mapped exactly the dispatcher plus three typed DSOs, and then calls
the central profile adapter.  Its result remains shadow/non-O5 because that is
the central profile's declared authority.
