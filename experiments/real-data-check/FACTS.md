# What the repository actually does

## Command and default

From the repository folder, the no-flags command in this workspace is:

```powershell
.\.venv312\Scripts\python.exe main.py
```

It does **not run a model**: it exits with code 2, because input
and output folders are required (src/unknown_dataset/runner.py:85-86).
The minimum usable form is:

```powershell
.\.venv312\Scripts\python.exe main.py --input PATH_TO_INPUT --output PATH_TO_OUTPUT
```

Omitting only the strategy chooses **baseline**. Exact source at
`src/unknown_dataset/runner.py:89`:

```python
parser.add_argument("--strategy", choices=("baseline", "auto", "routed"), default="baseline")
```

The default budget is 50 seconds and seed is 42 (src/unknown_dataset/runner.py:87,
88). The Python model API also defaults to baseline
(`src/unknown_dataset/models.py:19`).

## Published instructions versus code

The README explicitly says the official kickoff contract is unavailable
(`README.md:6`). Therefore **official organizer
compatibility cannot be certified**. The following checks concern the repo's
published, provisional instructions, not unprovided organizer rules.

| Item | Repository instructions | Actual implementation | Finding |
| --- | --- | --- | --- |
| Input files | README.md:65 | src/unknown_dataset/contract.py:50, 63, 109 | Matches: metadata.json, train.csv; supervised tasks also require test.csv. |
| Metadata | README.md:77 and 84 | src/unknown_dataset/contract.py:54, 100, 87 | Matches: task; supervised target; optional ID; exact feature type map; clustering k. |
| Hidden test labels | README.md:78 | src/unknown_dataset/contract.py:110 | Matches: test.csv containing target is rejected. |
| Labels/order | README.md:79 | src/unknown_dataset/contract.py:62, 118 | Matches: lexical class labels preserved; feature order aligned; row order retained. |
| Outputs | README.md:93-97 | src/unknown_dataset/contract.py:155, 160 | Matches: one prediction column plus profile.json and run.json. |
| Invalid input | README.md:99 | src/unknown_dataset/runner.py:226 | Nonzero errors are implemented. |

### Mismatches and qualifications

1. **No-flags execution is not supported.** The README's example supplies the
   required flags; there is no default input folder. Its `data/example` path is an
   example, not a bundled runnable dataset (existence here: False).
2. **Error wording is too broad.** README.md:99
   says model errors return nonzero. Routed can instead return successful recovered
   baseline predictions after optional errors (`src/unknown_dataset/routed.py:91`;
   `src/unknown_dataset/runner.py:190`). Earlier README paragraphs
   describe recovery correctly. Consumers must inspect run.json, not only exit code.
3. **Some documentation status is stale.** README.md:28
   describes CI as a remaining check; the approved portability checkpoint subsequently
   passed GitHub Actions. Old benchmark/test counts are historical, not current totals.
4. README.md:87
   should be read with the following sentence: numeric feature NaNs are allowed;
   infinities and missing targets are rejected (src/unknown_dataset/contract.py:43,
   77). This is ambiguous wording, not an observed format defect.

No instruction, contract, configuration or model file was edited for this check.
