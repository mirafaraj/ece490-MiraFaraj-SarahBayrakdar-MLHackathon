# Historical clustering input portability correction

Starting checkpoint: `6d82f9cd35af056a2908cf4a723a404ec0ee8c6a`.
No model experiment, dependency change, evaluator edit, commit or push is part of
this correction.

## Diagnosis before modification

The failed [GitHub Actions run](https://github.com/mirafaraj/ece490-MiraFaraj-SarahBayrakdar-MLHackathon/actions/runs/39396730781)
ran Linux/Python 3.12.15 and reported 216 passes / one failure. The first mismatch
was correlated / seed 6101 / feature missingness 30%:

- Original historical SHA-256:
  `34187feaf65a03508bdf3f8280f421dfc3c66cdba4c163f18525b3268ca19da0`.
- CI SHA-256:
  `82659b5ec836bea2b2dd171be7cba6bb5af40b9312c1ac52522325bfe8004b55`.

Before any code edits, the original test passed locally on Windows/Python 3.12.14.
Running that same unchanged test in a subprocess with
`OPENBLAS_CORETYPE=Haswell` reproduced the **exact CI hash and assertion failure**.
Forcing `SkylakeX` reproduced the historical hash. The local default dispatch is
SkylakeX, OpenBLAS 0.3.30. No dependency, source, serializer or Python change was
needed to reproduce the failure.

CI's installation log and the local environment agree on NumPy 2.3.5, pandas
3.0.1, SciPy 1.18.1 and scikit-learn 1.9.1. CPython's patch version differs, but
the controlled reproduction holds it fixed and changes only BLAS dispatch.

An isolated decomposition check found identical standard-normal draw hashes
across Haswell/SkylakeX, but different `numpy.linalg.svd` singular-value/factor
hashes and resulting matrix-product hashes. The correlated generator calls
`Generator.multivariate_normal`, which uses this linear algebra. This is CPU-
dependent floating-point generation, not a pandas/JSON/newline encoding issue.
NumPy explicitly documents that machine/build/LAPACK differences can affect
[multivariate-normal reproducibility](https://numpy.org/doc/2.3/reference/random/compatibility.html).

| Historical correlated seed | Differing finite values | Largest absolute difference |
| --- | ---: | ---: |
| 6101 | 94 | 4.440892098500626e-16 |
| 6103 | 63 | 8.881784197001252e-16 |
| 6107 | 92 | 8.881784197001252e-16 |

All eight historical cases retain identical missing-value masks and reference
labels under this intervention. The other five frames remain bit-identical.
The differing doubles are real data differences at roundoff scale; a cryptographic
hash correctly detects them. A fixed seed plus pinned Python packages is not an
exact-input archive.

## Small correction and retained guarantees

1. Added a 31,911-byte lossless fixture for the eight historical inputs. Numeric
   arrays use explicit little-endian float64, references int64, and columns Unicode;
   loading disables pickle. Every archived input was verified against the
   **existing** historical fingerprint, and every reference against the second-
   batch saved arrays before the fixture was written. Metadata records checkpoint,
   original report hash, generation environment, case identity and required k.
2. The second-batch offline harness loads exact archived inputs for historical
   replay, checks the archive checksum and each original frame fingerprint, and
   includes the fixture in future source-provenance maps. Fresh generation remains
   untouched. No rounding, generator changes or historical-hash rewriting occurs.
3. The previously failing test still requires eight losses and exact equality to
   **both original report hashes**. It separately checks regenerated column names,
   index, dtypes, missing masks, references and k exactly, and finite numeric values
   with `rtol=atol=1e-14`. This accommodates the measured sub-1e-15 drift without
   allowing changed masks, labels or meaningful feature changes. An added test
   requires rejection of a one-bit corrupted archive.

Fixture SHA-256:
`54cdc3d72271e2b452656ff0b0702883b1024f28ac8c8a4193b3c6ffa8fd3c8f`.

Only the offline replay harness and its tests are edited. The fixture and this
report/test XML are new. The original first-batch generator, fingerprints,
historical reports, source snapshots, production defaults, preprocessing,
selection and model implementations are preserved.

## Verification

- Before fix: original focused test passes with native SkylakeX; fails with
  Haswell using the same fingerprint as GitHub Actions.
- After fix: six focused clustering tests pass under native dispatch.
- Full repository suite with Haswell dispatch: **218 passed**, three expected
  warnings, 118.29s. The same kernel reproduced the original exact CI failure
  before the correction. See haswell-tests.xml.
- Native-SkylakeX correctness replay of the eight existing historical cases under
  both saved policies: **16/16 original prediction hashes, selected model names and selection
  reasons match**. This replays prior evidence without introducing candidates,
  tuning or new quality experiments. See historical-replay.json.
- Byte comparison against the checkpoint preserves all 15 production source
  files, three configuration files, 36 evaluator files, all 28 first/second-batch
  experiment artifacts and all 12 first/second-batch reports. Workflow and pinned
  dependencies are unchanged.
- Linux execution is unavailable locally: WSL is not installed and Docker is not
  available. The original CI log establishes the Linux failure; exact local
  reproduction and testing use its failing numerical kernel. This does not claim
  that a corrected Linux CI run has already occurred.

Suggested checkpoint after approval:
`fix: preserve exact historical clustering inputs across platforms`.
