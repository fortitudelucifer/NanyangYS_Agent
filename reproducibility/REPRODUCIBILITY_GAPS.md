# Reproducibility gaps and required disclosures

The repository provides enough material to reconstruct and audit the pipeline,
but it does not claim that a new machine will produce bitwise-identical model
outputs without additional controls.

## Known gaps

1. The exact NVIDIA driver version for the v010-v012 RTX 5070 Ti release runs
   was not recorded. The manifests record Torch 2.11.0+cu128, compiled CUDA
   12.8, compute capability 12.0, and GPU memory, but not the driver.
2. The reference package records SHA-256 values for the extracted CSV files,
   not a trusted SHA-256 for the downloaded `KuaiRand-1K.tar.gz` transport.
3. GPU kernels, BLAS/thread scheduling, operating system, filesystem ordering,
   and hardware generation can prevent bitwise equality even with the same
   Python packages and seeds.
4. Historical approval JSON files document the original controlled execution.
   They are not transferable permission for a new experiment.
5. Later experiments depend on hash-verified predecessor artifacts. The repo
   intentionally omits those large artifacts, so they must be rebuilt in order.

## Minimum record for a new attempt

- repository commit and dirty/clean state;
- download URL, archive byte size, and archive SHA-256;
- extracted Raw verification result;
- OS, kernel/build, Python executable and version;
- complete `pip freeze` or equivalent environment lock;
- CPU, RAM, GPU, NVIDIA driver, `nvidia-smi`, CUDA build and compute capability;
- all model/fit/bootstrap seeds and thread-count environment variables;
- start/end timestamps, elapsed time, and stage access ledger;
- output row counts, SHA-256 manifests, quality gates, and test results.

Any mismatch must be reported as a separate reproduction attempt. Do not edit
the frozen expected manifests to make a new run appear identical.
