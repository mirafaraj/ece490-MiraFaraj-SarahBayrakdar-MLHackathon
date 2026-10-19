# Independent supervised experiments

Development seed: 42. One outer held-out split per dataset (25%). Candidate selection uses a separate holdout inside the training split; outer labels are used only for this report. Bundled real datasets require no downloads. Classification metric is balanced accuracy (higher better); regression is RMSE (lower better). Timing measures in-process model work, excluding CLI imports and worker startup. Results are development evidence, not confirmation or official contest scores.

| Dataset | Task | Baseline | Auto | Baseline seconds | Auto seconds | Selected |
|---|---|---:|---:|---:|---:|---|
| iris | classification | 0.92308 | 0.92308 | 0.281 | 0.328 | linear |
| wine | classification | 1.00000 | 1.00000 | 0.219 | 0.531 | extra_trees |
| breast_cancer | classification | 0.95115 | 0.96834 | 0.203 | 2.766 | linear |
| diabetes | regression | 51.72851 | 53.31824 | 0.203 | 0.234 | linear |
| linear | regression | 72.40271 | 1.83926 | 0.390 | 0.375 | linear |
| friedman1 | regression | 1.94669 | 1.42336 | 0.375 | 0.469 | histogram |
| imbalanced | classification | 0.77778 | 0.84737 | 0.219 | 2.531 | histogram |

Seven focused supervised regression tests pass. Selection is optional and provisional: fixed metrics may differ from contest scoring; one holdout has sampling uncertainty; selection may regress on unseen tables. Dense histogram fits cap at 2 million cells; prediction batches honor that cap. Candidate time checks cannot interrupt an estimator, so the runner must enforce the process deadline. Rare/tiny/low-budget and very large inputs skip selection. Full-data histogram overflow retries Extra Trees. Only independently held-out evidence should promote the default strategy.
