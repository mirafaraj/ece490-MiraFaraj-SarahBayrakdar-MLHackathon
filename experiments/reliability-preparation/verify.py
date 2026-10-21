"""Serial CLI verification; immutable historical inputs and separate new outputs."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/real-data-check'
sys.path.insert(0, str(HERE))
from profile_reference import reconstruct, compare


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [ROOT/'main.py',ROOT/'README.md',ROOT/'MODEL_CARD.md',ROOT/'requirements.lock',ROOT/'pyproject.toml']
    paths += list((ROOT/'src').rglob('*.py')) + list((ROOT/'configs').glob('*.json'))
    paths += list((ROOT/'tests').glob('*.py')) + list(HERE.glob('*.py')) + list((HERE/'trace').glob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}


def check_freeze():
    path = HERE/'freeze.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        assert source_hashes() == frozen['source_sha256'], 'Frozen files changed: STOP'
        assert digest(HERE/'frozen-source.zip') == frozen['archive_sha256'], 'Frozen archive changed: STOP'


def profile(case, output):
    cache=HERE/'profile-reference'/f'{case["id"]}.json'
    if cache.exists():
        expected=json.loads(cache.read_text())
    else:
        expected=reconstruct(OLD/case['input'])
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(expected,indent=2)+'\n')
    return compare(expected,json.loads((output/'profile.json').read_text()))


def plans(section, cases):
    stress=[c for c in cases if c['group']=='stress']
    real=[c for c in cases if c['group']=='real']
    if section=='nine':
        for c in stress:
            if c['task']=='regression' and c['dataset'].startswith('shape_'):
                for s in ['baseline','auto','routed']:yield c,s,0
            elif c['task']=='classification' and c['dataset']=='shape_100000x50':
                for s in ['baseline','routed']:yield c,s,0
            elif c['task']=='classification' and c['dataset']=='shape_5000x500':yield c,'auto',0
    elif section in ['stress','real']:
        for c in stress if section=='stress' else real:
            for s in ['baseline','auto','routed']:yield c,s,0
    elif section=='submission':
        for c in cases:yield c,'submission',0
    elif section=='final':
        for c in stress:yield c,'submission',0
        for repeat in [1,2]:
            for c in stress:
                if c['task']!='clustering' and c['dataset'].startswith('shape_'):yield c,'submission',repeat
        for c in real:yield c,'submission',0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('section',choices=['nine','stress','real','submission','final','profiles'])
    args=parser.parse_args()
    check_freeze()
    import numpy as np
    import pandas as pd
    spec=importlib.util.spec_from_file_location('historical_real_check',OLD/'check.py')
    scoring=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    manifest=json.loads((OLD/'manifest.json').read_text())
    oldrows={(r['case_id'],r['strategy']):r for r in scoring.existing_rows()}
    if args.section=='profiles':
        audit=[]
        for c in manifest['cases']:
            if c['group']!='stress':continue
            for s in ['baseline','auto','routed']:
                output=OLD/'runs'/c['id']/s
                if (output/'profile.json').exists():
                    audit.append(dict(case_id=c['id'],strategy=s,fields=profile(c,output)))
        (HERE/'profile-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print('FULL-DATA PROFILE AUDIT',len(audit),'saved profiles',len(set(r['case_id'] for r in audit)),'inputs',flush=True)
        return
    record=HERE/f'{args.section}.jsonl'
    done=[json.loads(l) for l in record.read_text().splitlines()] if record.exists() else []
    for c,s,repeat in plans(args.section,manifest['cases']):
        if any(r['case_id']==c['id'] and r['strategy']==s and r['repeat']==repeat for r in done):continue
        check_freeze()
        for name,expected in c['input_sha256'].items():assert digest(OLD/c['input']/name)==expected
        assert digest(OLD/c['reference'])==c['reference_sha256']
        directory=HERE/args.section/c['id']/f'{s}-{repeat}'
        directory.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-B',str(ROOT/'main.py'),'--input',str(OLD/c['input']),
                 '--output',str(directory),'--strategy',s,'--time-budget','50','--seed',str(c['seed'])]
        # Final phase deliberately uses the exact documented submission command;
        # instrumentation is confined to initial repair measurements.
        if args.section=='nine':command.append('--timings')
        start=time.monotonic()
        with (directory/'stdout.txt').open('w') as out,(directory/'stderr.txt').open('w') as err:
            child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                child.wait(timeout=10)
        row=dict(case_id=c['id'],task=c['task'],strategy=s,repeat=repeat,
                 seconds=time.monotonic()-start,returncode=child.returncode,valid=False)
        try:
            assert child.returncode==0,(directory/'stderr.txt').read_text()
            reference=json.loads((OLD/c['reference']).read_text())
            table=pd.read_csv(directory/'predictions.csv',dtype=str)
            assert list(table.columns)==['prediction'] and len(table)==len(reference['truth']) and not table.prediction.isna().any()
            if c['task']=='classification':
                prediction=table.prediction.tolist()
                assert set(prediction)<=set(reference['training_labels'])
            else:
                prediction=table.prediction.astype(float).to_numpy()
                assert np.isfinite(prediction).all()
                if c['task']=='clustering':
                    assert np.equal(prediction,np.floor(prediction)).all()
                    prediction=prediction.astype(int)
                    assert set(prediction)==set(range(c['k']))
            run=json.loads((directory/'run.json').read_text())
            fields=profile(c,directory)
            model=run['model']
            selected=model.get('selected_candidate',model.get('model'))
            sha=digest(directory/'predictions.csv')
            row.update(valid=True,profile_fields=fields,prediction_sha256=sha,
                       model=model,metrics=scoring.scores(c['task'],reference['truth'],prediction,reference['training_labels']))
            # Historical prediction equivalence is mandatory only for ordinary-size cases.
            effective=('baseline' if c['task']=='classification' else 'auto') if s=='submission' else s
            if c['group']=='real':
                original=oldrows[(c['id'],effective)]
                assert sha==original['prediction_sha256'],'STOP: normal-size predictions changed'
                assert selected==original['selected_candidate'],'STOP: normal-size selected model changed'
                row['historical_equivalent']=True
            if s=='submission' and c['group']=='stress':
                counterpart=HERE/'stress'/c['id']/f'{effective}-0'/'predictions.csv'
                assert digest(counterpart)==sha,'Submission delegation prediction mismatch'
                row['delegation_equivalent']=True
            assert row['seconds']<60,'Execution limit breached'
            if args.section=='nine':assert not model['model'].startswith('Emergency'),'Emergency recovery: STOP'
        except Exception as exc:
            row['error']=str(exc)
        with record.open('a') as handle:handle.write(json.dumps(row)+'\n')
        print(json.dumps({k:row.get(k) for k in ['case_id','strategy','repeat','seconds','valid','error']}),flush=True)
        check_freeze()
        if row.get('error'):raise RuntimeError(row['error'])


if __name__=='__main__':main()
