"""Concise tables without reference or prediction arrays."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    report = json.loads((HERE / "local/results.json").read_text())
    rows = pd.DataFrame(report["rows"])
    reference = rows[rows.ari.notna()].copy()
    # Seeds within a condition, conditions within a family, then equal family weight.
    means = reference.groupby(["family", "condition", "policy"])[["ari", "ami"]].mean()
    family_means = means.groupby(["family", "policy"]).mean()
    aggregate = family_means.groupby("policy").mean()
    base = reference[reference.policy == "full"].set_index("id")
    pairs = []
    for _, row in reference.iterrows():
        pairs.append(dict(id=row.id, family=row.family, condition=row.condition, policy=row.policy,
            ari=row.ari, ami=row.ami, delta_ari=row.ari-base.loc[row.id].ari,
            delta_ami=row.ami-base.loc[row.id].ami, smallest_group_recall=row.smallest_group_recall,
            group_sizes=row.group_sizes, repairs=row.repairs, selected=row["info"]["model"],
            direct_seconds=row.direct_seconds, valid=row.valid, fallback=row.fallback))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(HERE / "per-case-metrics.csv", index=False)
    mall = rows[rows.family == "Mall"]
    mall[["id", "policy", "silhouette_common", "calinski_harabasz_common", "davies_bouldin_common",
          "mean_partition_stability", "group_sizes", "repairs", "direct_seconds", "valid"]].to_csv(HERE / "mall-metrics.csv", index=False)
    lines = ["# Per-case clustering recovery (diagnostic development evidence)", "",
        "Rows share identical frozen inputs, k and fitting seeds. ARI / AMI; changes are versus full strength.", "",
        "| Case | Full ARI / AMI | Quarter ARI / AMI (delta ARI) | Omitted ARI / AMI (delta ARI) |",
        "|---|---:|---:|---:|"]
    for identity in base.index:
        case = pairs[pairs.id == identity].set_index("policy")
        cell = lambda p: f"{case.loc[p].ari:.4f} / {case.loc[p].ami:.4f}"
        lines.append(f"| {identity} | {cell('full')} | {cell('quarter')} ({case.loc['quarter'].delta_ari:+.4f}) | {cell('omitted')} ({case.loc['omitted'].delta_ari:+.4f}) |")
    (HERE / "PER_CASE.md").write_text("\n".join(lines) + "\n")
    summary = dict(main_calls=len(rows), stability_calls=30, all_valid=bool(rows.valid.all()),
        fallback_count=int(rows.fallback.sum()), repairs=rows.groupby("policy").repairs.sum().to_dict(),
        family_balanced=aggregate.to_dict(orient="index"),
        strata=means.reset_index().to_dict(orient="records"),
        max_direct_seconds=rows.groupby("policy").direct_seconds.max().to_dict(),
        worst={policy: pairs[pairs.policy == policy].sort_values("delta_ari").iloc[0].to_dict()
               for policy in ["quarter", "omitted"]},
        severe_ari_losses=pairs[pairs.delta_ari < -.10].to_dict(orient="records"),
        decision="Retain incumbent full strength. Neither alternative passes registered gates; no confirmation opened.")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(aggregate.round(5).to_string())
    for policy in ["quarter", "omitted"]:
        worst = summary["worst"][policy]
        print(policy, "worst", worst["id"], worst["delta_ari"])


if __name__ == "__main__":
    main()
