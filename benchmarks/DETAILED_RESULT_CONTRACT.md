# Detailed Result Contract

Every evaluator used for a final comparison must write the legacy
`standard_metrics.json` and the detailed contract below.  A method with a
non-`passed` `benchmark_report.json` is **incomplete**, even when stale metric
files remain in its output directory. A process-level `passed` report is also
incomplete when the detailed manifest contains one or more failed sequence
records; recovered sequence failures must not be hidden by aggregate metrics.
The runner report is preferred for command/exit-time provenance.  If it is
missing, a directory with complete standard metrics, a complete detailed
manifest, zero failed records, and all referenced arrays is classified as
`direct-artifacts`: it may be used for numerical analysis, but the provenance
warning must remain visible and the run is not reproducibility-complete.

## Required files

- `detailed_metrics.jsonl`: one JSON object per attempted sequence.
- `detailed_metrics_manifest.json`: method, suite, FPS and record count.
- `detailed_metrics/sequence_*.npz`: required for every passed record, with
  `rotation_deg[T,24]`, `translation_m[T]` when available, and `fps`.

Each JSONL record requires `source`, `action`, `configuration`, `status`,
`failure_reason`, `valid_frames`, rotation summaries, root-translation RMSE,
endpoint error, drift rate, contact foot-sliding proxy, and `array_file`.
Action labels must preserve `lying`, `crawling`, `transitions`, `interaction`,
and `sports`; absent categories are reported as missing, not silently dropped.

For MobilePoser, the physical sensor count includes the pelvis reference slot:
4 IMUs = model slots `[0,1,2]` plus pelvis slot `5`; 5 IMUs = `[0,1,2,3]`
plus slot `5`; 6 IMUs = `[0,1,2,3,4]` plus slot `5`.  The network input remains
60-D in all three cases, so the 6-IMU row is not a six-input architecture.

Use `python benchmarks/summarize_detailed.py --results-root benchmark_results`
after a server run.  Its output has mean, P90, and failure-rate rows per method,
configuration, and action.

`summarize_standard.py` applies the same manifest/report gate and emits
`eligible_for_ranking` plus `failed_sequences_total`; do not sort its rows before
filtering on that flag.

Before building a paper table, run
`python benchmarks/validate_results.py --results-root benchmark_results`.
Any method reported as `INCOMPLETE` or `DIAGNOSTIC-ONLY` must remain out of the
unconditional ranking.  `DIAGNOSTIC-ONLY` results retain successful-sequence
metrics and must report the failed-sequence count/rate.  A `WARNING ... direct
metric artifacts are complete` line does not require a re-run for numerical
metrics, but should be resolved before a final artifact release if runner
provenance is required.
