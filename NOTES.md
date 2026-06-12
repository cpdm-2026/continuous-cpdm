# CPDM Implementation Notes

This document summarizes implementation details for the modular CPDM codebase.

## Important notes

1. `SpinorConditionEncoder` was intentionally excluded because the current `UNetDenoiser` directly consumes scalar `s_z` through `spin_mlp`.
2. Notebook-only code (interactive mount calls, inline dataset download commands) was not included in the importable modules.
3. The original top-level leaf-copy block was wrapped as `prepare_leaf_subset()` in `data.py`.
4. `train_alt()` is kept mostly intact because its nested `@tf.function train_step` closes over TensorFlow tensors and optimizer state.
5. Path constants are temporary defaults in `config.py`; these should become CLI/config-file values later.
6. `build_and_load_latest()` now accepts optional datasets. If only sampling, make sure `PROTO_PATH` already exists.

7. `train_alt()` now creates output/checkpoint/prototype directories before training.
