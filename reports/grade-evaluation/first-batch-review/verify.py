"""Independent saved-evidence verification only; no production imports or fits."""
from collections import Counter, defaultdict
import ast
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assessment import family_interval, quality_outcome, rank_regret
from promotion import evaluate

METRICS=("accuracy","balanced_accuracy","macro_f1")
checks=0
inputs={}


def check(ok, message):
    global checks
    checks+=1
    if not ok: raise AssertionError(message)


def close(actual, expected, label):
    check(abs(actual-expected)<1e-10,label)


def load(name):
    path=ROOT/'experiments/first-batch'/name
    raw=path.read_bytes()
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def policy_key(row): return row['holdouts'],row['weighting'],row['objective']


def verify_classification():
    study=load('classification.json'); known=load('classification-known8081.json')
    all_cases=study['cases']+[study['known_inspected_case'],known['case']]
    preference={'linear':0,'extra_trees':1,'histogram':2}
    for case in all_cases:
        candidates=case['candidates']
        check(len(candidates)==6,'six fixed candidate corners')
        for candidate in candidates.values():
            check(candidate['valid'] and not candidate['fallback'],'candidate valid without fallback')
            check(len(candidate['inner'])==2,'two recorded folds')
        for choice in case['selections']:
            pool=[k for k in candidates if choice['weighting']=='both' or k.endswith(':'+choice['weighting'])]
            scores={k:mean(f['metrics'][choice['objective']] for f in candidates[k]['inner'][:choice['holdouts']]) for k in pool}
            selected=min(pool,key=lambda k:(-scores[k],preference[candidates[k]['family']],pool.index(k)))
            check(selected==choice['selected'],'selection reconstructed from inner scores only')
            check(choice['outer']==candidates[selected]['outer'],'outer belongs to selected candidate')
            for key,value in scores.items(): close(choice['validation_scores'][key],value,'saved fold means')
            for metric in METRICS:
                close(choice['gaps_to_best_evaluated'][metric],max(candidates[k]['outer'][metric] for k in pool)-choice['outer'][metric],'retrospective gap')
                close(choice['losses_vs_baseline'][metric],candidates['extra_trees:balanced']['outer'][metric]-choice['outer'][metric],'baseline loss')
    for summary in study['summary']:
        rows=[c for case in study['cases'] for c in case['selections'] if policy_key(c)==policy_key(summary)]
        check(dict(Counter(r['selected'] for r in rows))==summary['selection_frequency'],'selection frequency')
        for metric in METRICS:
            close(summary['mean'][metric],mean(r['outer'][metric] for r in rows),'policy mean')
            close(summary['mean_gap'][metric],mean(r['gaps_to_best_evaluated'][metric] for r in rows),'mean gap')
            close(summary['worst_loss_vs_baseline'][metric],max(r['losses_vs_baseline'][metric] for r in rows),'worst loss')
    changes=[]
    severe=[]
    for case in study['cases']:
        one=next(r for r in case['selections'] if policy_key(r)==(1,'balanced','balanced_accuracy'))
        two=next(r for r in case['selections'] if policy_key(r)==(2,'balanced','balanced_accuracy'))
        changes.append(dict(case=case['case'],one=one['selected'],two=two['selected'],
                            **{m:two['outer'][m]-one['outer'][m] for m in METRICS}))
        for metric in METRICS:
            delta=two['outer'][metric]-case['candidates']['extra_trees:balanced']['outer'][metric]
            if delta<=-.10+1e-12: severe.append(dict(case=case['case'],metric=metric,change=delta,selected=two['selected'],definition=case['definition']))
    source=(ROOT/'src/unknown_dataset/supervised.py').read_bytes()
    reconstructed=source.replace(b'if name == "extra_trees" and completed_baseline is None:',b'if name == "extra_trees":')
    check(hashlib.sha256(reconstructed).hexdigest()==study['source_sha256'],'exact classification pre-refinement source reconstruction')
    mismatches={p:h for p,h in study['source_hashes'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h}
    check(set(mismatches)=={'src\\unknown_dataset\\supervised.py'},'only documented study production-source difference')
    corners={k:v['outer'] for k,v in known['case']['candidates'].items()}
    b=corners['linear:balanced']['balanced_accuracy']; u=corners['linear:unweighted']['balanced_accuracy']; t=corners['extra_trees:unweighted']['balanced_accuracy']
    return dict(cases=12,policy_summaries_verified=len(study['summary']),candidate_fits_main_and_known=78,
        holdout_changes=changes,holdout_mean_changes={m:mean(c[m] for c in changes) for m in METRICS},
        holdout_outcomes={m:dict(Counter('win' if c[m]>1e-12 else 'loss' if c[m]<-1e-12 else 'tie' for c in changes)) for m in METRICS},
        severe_two_holdout_vs_baseline=severe,corners= corners,
        path_ba_losses=dict(weighting=b-u,family_selection=u-t,total=b-t),
        repeat_flags_recorded=study['repeats'],
        limitations=['One dataset/model seed for rotating-factor cases; not twelve independent families.',
                    'No per-case input hashes or raw truth/prediction arrays in the classification study; arithmetic/selection verified, not raw metric recomputation.',
                    'Exhaustive offline scores are not budgeted selector deployment evidence; repeats store verdicts rather than complete second-run evidence.'])


def verify_clustering():
    data=load('clustering.json'); diagnostic=load('clustering-diagnostic.json')
    indexed={}
    for row in data['rows']:
        key=(row['family'],row['mechanism'],row['fraction'],row['seed'],row['policy'])
        check(key not in indexed,'unique clustering case/policy')
        indexed[key]=row
        check(row['valid'],'valid clustering output')
    check(len(indexed)==360,'complete clustering count')
    summaries={}
    for policy in data['policies']:
        rows=[r for r in data['rows'] if r['policy']==policy]
        losses=[]; changes=[]; family_values=defaultdict(lambda:defaultdict(list))
        for row in rows:
            before=indexed[(row['family'],row['mechanism'],row['fraction'],row['seed'],'current')]
            check(row['input_sha256']==before['input_sha256'],'paired cluster input hash')
            delta={m:row[m]-before[m] for m in ('ari','ami')}
            changes.append(delta)
            for m in delta: family_values[row['family']][m].append(delta[m])
            if delta['ari'] < -1e-12: losses.append(dict(family=row['family'],mechanism=row['mechanism'],fraction=row['fraction'],seed=row['seed'],ari_change=delta['ari'],ami_change=delta['ami'],model=row['info']['model']))
        means={f:{m:mean(v) for m,v in metrics.items()} for f,metrics in family_values.items()}
        summaries[policy]=dict(mean_ari=mean(r['ari'] for r in rows),mean_ami=mean(r['ami'] for r in rows),
            ari_outcomes=dict(Counter('win' if d['ari']>1e-12 else 'loss' if d['ari']< -1e-12 else 'tie' for d in changes)),
            ami_outcomes=dict(Counter('win' if d['ami']>1e-12 else 'loss' if d['ami']< -1e-12 else 'tie' for d in changes)),
            family_improvements=means,exploratory_family_interval95={m:family_interval([v[m] for v in means.values()]) for m in ('ari','ami')},
            severe_ari_losses=[r for r in losses if r['ari_change'] <= -.10+1e-12],all_ari_losses=sorted(losses,key=lambda r:r['ari_change']))
    for p,h in data['source_sha256'].items(): check(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,'clustering source hash')
    repeats=[r for r in diagnostic['rows'] if r['kind']=='repeat']
    check(len(repeats)==20 and all(r['identical'] for r in repeats),'twenty recorded repeated configurations')
    return dict(rows=len(indexed),families=6,policies=summaries,
        limitations=['Six related synthetic mechanisms, not ninety independent datasets.',
                    'Fixed policy execution order and metric bookkeeping prevent causal speed claims.',
                    'Feature-dependent masks do not establish safety under genuinely label-informative missingness.',
                    'Direct calls cannot certify full CLI deadlines or eight-family P0 promotion criteria.'])


def verify_runtime():
    combined=load('runtime-combined.json')
    segments=[load(n) for n in ('runtime-study.json','runtime-supplement.json','runtime-supplement-2.json')]
    key=lambda r:(r['shape'],r['budget'],r['repeat'],r['strategy'])
    raw=[r for s in segments for r in s['rows']]
    check(len(raw)==len({key(r) for r in raw})==54,'runtime union unique54')
    check({key(r) for r in raw}=={key(r) for r in combined['planned']},'runtime planned inventory')
    check({key(r):r for r in raw}=={key(r):r for r in combined['rows']},'combined runtime equals exact source union')
    production_maps=[{p:h for p,h in s['source_sha256'].items() if not p.startswith('scripts/')} for s in segments]
    check(all(m==production_maps[0] for m in production_maps),'production and configuration identities stable across timing segments')
    for p,h in production_maps[0].items(): check(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,'current production matches runtime source')
    check(all(s['fixture_sha256']==combined['fixture_sha256'] for s in segments),'fixture bytes match across segments')
    grouped=defaultdict(list)
    for row in raw:
        grouped[(row['shape'],row['budget'],row['strategy'])].append(row)
        check(row['valid'] and row['returncode']==0 and not row['external_timeout'],'runtime actual CLI success')
    for summary in combined['summary']:
        rows=grouped[(summary['shape'],summary['budget'],summary['strategy'])]
        check(len(rows)==summary['runs']==3,'three repetitions per cell')
        close(mean(r['external_seconds'] for r in rows),summary['mean_external_seconds'],'mean external runtime')
        for metric in (*METRICS,'minority_recall'):
            close(mean(r[metric] for r in rows),summary['mean_'+metric],'runtime quality summary')
        check(sorted({r['prediction_sha256'] for r in rows})==summary['repeat_prediction_hashes'],'runtime repeat hash counts')
    canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    native_rows=[]
    for row in raw:
        model=row['run']['model']
        native_rows.append(dict(task='classification',family=row['shape'],name=f"{row['shape']}:budget{row['budget']}:repeat{row['repeat']}",seed=42,
            strategy=row['strategy'],valid=row['valid'],exit_code=row['returncode'],error=None,partition='development',
            data_sha256=canonical(combined['fixture_sha256'][row['shape']]),wall_seconds=row['external_seconds'],
            fallback=bool(model.get('fallback',False)),run_info=row['run'],
            metrics={**{m:row[m] for m in METRICS},'recall_by_class':{'minority':row['minority_recall']}}))
    source=canonical(production_maps[0])
    native=dict(config=dict(source_sha256=source,source_sha256_after=source,source_unchanged=True,timeout_seconds=60,
                           seeds=[42],strategies=['baseline','auto','routed']),datasets=native_rows)
    assessment=evaluate(native,candidate='routed',baseline='auto',manifest={'stage':'development','comparison_pool':['baseline','auto','routed']})
    return dict(rows=54,source_segment_sizes=[len(s['rows']) for s in segments],max_external_seconds=max(r['external_seconds'] for r in raw),
        completed_recoveries=sum(r['run']['model'].get('fallback_kind')=='completed_baseline' for r in raw),
        emergency_fallbacks=sum(bool(r['run']['model'].get('fallback')) and r['run']['model'].get('fallback_kind')!='completed_baseline' for r in raw),
        rows_with_memory=sum(r['peak_process_tree_rss_bytes'] is not None for r in raw),summary=combined['summary'],p0_assessment=assessment,
        adapter_note='Derived from verified actual CLI rows; budgets/repeats are nested within two fixture families, not fresh seeds. Canonical hash of verified production map is adapter identity, not official source fingerprint.',
        controlled_recovery=combined['controlled_recovery'])


def main():
    output=Path(__file__).with_name('evidence.json')
    if output.exists(): raise SystemExit('Choose a new output location; existing evidence is immutable')
    classification=verify_classification(); clustering=verify_clustering(); runtime=verify_runtime()
    defaults=next(ast.literal_eval(n.value) for n in ast.parse((ROOT/'src/unknown_dataset/policies.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_CONFIG' for t in n.targets))
    result=dict(scope='Independent saved-evidence audit; no fits, network, production imports or configuration writes.',
        classification=classification,clustering=clustering,runtime=runtime,
        defaults=dict(file=json.loads((ROOT/'configs/router.json').read_text()),experimental=defaults['experimental']),
        checks_passed=checks,input_sha256=inputs)
    equivalent=load('default-equivalence.json')
    check(equivalent['frozen']==equivalent['current'] and len(equivalent['current'])==18,'eighteen frozen default equivalence records')
    xml=ET.parse(ROOT/'experiments/first-batch/integration-tests.xml')
    result['recorded_test_evidence']=[dict(s.attrib) for s in xml.findall('.//testsuite')]
    result['checks_passed']=checks
    with output.open('x',encoding='utf-8') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({'checks_passed':checks,'output':str(output),'classification_holdout_changes':classification['holdout_mean_changes'],
                      'clustering':{p:{k:r[k] for k in ('mean_ari','mean_ami','ari_outcomes','ami_outcomes')} for p,r in clustering['policies'].items()},
                      'runtime':{k:runtime[k] for k in ('rows','max_external_seconds','completed_recoveries','emergency_fallbacks')}},indent=2))


if __name__=='__main__': main()
