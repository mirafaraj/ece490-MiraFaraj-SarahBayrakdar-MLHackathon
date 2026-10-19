"""Pair saved frozen versions without executing either model."""
from copy import deepcopy
from assessment import obj, sha


def pair_reference(report, reference, reference_hash, reference_label, run_strategy, candidate):
    if reference_label == candidate:
        raise ValueError("Reference alias must differ from candidate strategy")
    for data in (report, reference):
        config = obj(obj(data).get("config"))
        if not sha(config.get("source_sha256")) or config.get("source_unchanged") is not True or config.get("source_sha256_after") != config.get("source_sha256"):
            raise ValueError("Both version reports must have completed frozen provenance")
    if not sha(reference_hash):
        raise ValueError("Reference report hash is required")
    result=deepcopy(report)
    reference_rows=[r for r in reference.get("datasets",[]) if r.get("strategy")==run_strategy]
    candidate_rows=[r for r in report.get("datasets",[]) if r.get("strategy")==candidate]
    if not reference_rows or not candidate_rows:
        raise ValueError("Requested candidate/reference run is absent")
    def indexed(rows):
        index={}
        for row in rows:
            key=(row["task"],row["name"],row["seed"])
            if key in index: raise ValueError("Duplicate version comparison case")
            index[key]=row
        return index
    ref_index, candidate_index=indexed(reference_rows), indexed(candidate_rows)
    if set(ref_index)!=set(candidate_index):
        raise ValueError("Version comparisons require exactly identical case inventories")
    for key,row in candidate_index.items():
        if row.get("data_sha256")!=ref_index[key].get("data_sha256") or not sha(row.get("data_sha256")):
            raise ValueError("Version comparison input hashes differ")
    result["datasets"]=[deepcopy(r) for r in candidate_rows]
    for original in reference_rows:
        row=deepcopy(original)
        row["strategy"]=reference_label
        result["datasets"].append(row)
    result["config"]["strategies"]=[reference_label,candidate]
    result["config"]["strategy_versions"]={
        reference_label:dict(strategy=run_strategy,source_sha256=reference["config"]["source_sha256"],
                            config_sha256=reference["config"].get("router_config_sha256"),report_sha256=reference_hash),
        candidate:dict(strategy=candidate,source_sha256=report["config"]["source_sha256"],
                       config_sha256=report["config"].get("router_config_sha256"))}
    return result
