# v013 implementation revision 02

- Formal model run completed with status `complete`: 20/20 curve points, 40/40 GPU fits and zero point failures.
- Post-run visual QA found that the A supporting figure used each metric row's own `rows` field as x. Validation has a fixed 886,452 rows, so its five points overlapped at one x coordinate.
- Plot-only correction: map every A train/Validation metric row to the corresponding `n_train_events` recorded in `history_uplift.csv`.
- No prediction, model, calibration, point metric, bootstrap replicate, interval or decision value changed.
- The corrected figure and expanded Markdown interpretation are covered by a refreshed output artifact hash manifest. The original result-producing runner hash remains preserved in `outputs/run_manifest.json`.
