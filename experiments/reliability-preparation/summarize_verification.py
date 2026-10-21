"""Summarize saved verification records; never fits models or changes sources."""
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import json
import statistics

from verify import HERE,ROOT,OLD,check_freeze


def main():
    check_freeze()
    spec=importlib.util.spec_from_file_location('old_scorer',OLD/'check.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    originals={(r['case_id'],r['strategy']):r for r in scorer.existing_rows()}
    manifest=json.loads((OLD/'manifest.json').read_text())
    cases={c['id']:c for c in manifest['cases']}
    summary={}
    for name in ['nine','stress','real','submission','final']:
        path=HERE/f'{name}.jsonl'
        if not path.exists():continue
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        summary[name]=dict(calls=len(rows),valid=sum(r['valid'] for r in rows),
            maximum_seconds=max(r['seconds'] for r in rows),
            over_45=[dict(case_id=r['case_id'],strategy=r['strategy'],seconds=r['seconds']) for r in rows if r['seconds']>45],
            emergency_outputs=sum(r.get('model',{}).get('model','').startswith('Emergency') for r in rows),
            linear_recoveries=sum(r.get('model',{}).get('fallback_kind')=='bounded_linear' for r in rows),
            historical_equivalence_count=sum(r.get('historical_equivalent',False) for r in rows),
            delegation_equivalence_count=sum(r.get('delegation_equivalent',False) for r in rows),
            selected_models=dict(Counter(r.get('model',{}).get('model','unavailable') for r in rows)))
        if name=='stress':
            improvements=[]
            losses=[]
            for r in rows:
                c=cases[r['case_id']]
                if c['task']!='clustering':
                    ref=json.loads((OLD/c['reference']).read_text())
                    train=ref['training_labels']
                    if c['task']=='regression':
                        import pandas as pd
                        meta=json.loads((OLD/c['input']/'metadata.json').read_text())
                        train=pd.read_csv(OLD/c['input']/'train.csv',usecols=[meta['target']])[meta['target']].tolist()
                        dummy=[statistics.mean(train)]*len(ref['truth'])
                    else:
                        label=sorted(Counter(train).items(),key=lambda item:(-item[1],item[0]))[0][0]
                        dummy=[label]*len(ref['truth'])
                    metrics=scorer.scores(c['task'],ref['truth'],dummy,train)
                    improvements.append(dict(case_id=c['id'],strategy=r['strategy'],
                        dummy_metrics=metrics,learned_metrics=r['metrics']))
                old=originals[(c['id'],r['strategy'])]
                if old['valid'] and not old.get('fallback_kind'):
                    for metric,value in r['metrics'].items():
                        if isinstance(value,(int,float)) and metric in old['metrics']:
                            gain=old['metrics'][metric]-value if metric in ['rmse','mae'] else value-old['metrics'][metric]
                            if gain < -1e-12:losses.append(dict(case_id=c['id'],strategy=r['strategy'],metric=metric,gain=gain))
            summary[name]['dummy_comparisons']=improvements
            summary[name]['losses_vs_successful_originals']=losses
    result=HERE/'verification-summary.json'
    result.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x not in ['dummy_comparisons','losses_vs_successful_originals']} for k,v in summary.items()},indent=2))
    check_freeze()


if __name__=='__main__':main()
