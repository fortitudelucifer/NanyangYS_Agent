# v013 implementation revision 01

- Contract SHA-256 remains `0e67c6d5388fba55157fd5f2af7d0681a55e280b96ebed492a9d3c77ac522c24`.
- First release attempt was manually interrupted before any curve-point result was written. Its preflight and bootstrap multiplicity artifacts are preserved in `attempt_01_interrupted_bootstrap_block8/`.
- Observed issue: the inherited weighted-AP bootstrap block size of 8 kept the first curve point in CPU bootstrap for more than nine minutes.
- Implementation-only change: increase `AP_BOOTSTRAP_BLOCK_SIZE` from 8 to 64.
- Unchanged scientific semantics: 2,000 bootstrap replicates, seed, multiplicity matrix construction, user cluster unit, paired BL1/BL2 weights, point estimates, percentile intervals, training data, preprocessing, models and optimizer.
- Expected effect: fewer Python loop iterations and higher temporary vectorization memory, with identical arithmetic definition up to ordinary floating-point execution order.
- Re-approval assessment: no contract field or scientific/protocol parameter changed; the existing exact contract approval remains applicable. The final run manifest records the revised runner SHA-256.
