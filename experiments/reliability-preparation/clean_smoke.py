"""Three explicit submission CLI smoke checks under the isolated pinned environment."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

from verify import HERE,ROOT,OLD,check_freeze,profile


def main():
    check_freeze()
    assert (HERE/'freeze.json').exists(),'Final clean smoke must follow freeze'
    import importlib.metadata
    import numpy as np
    import pandas as pd
    environment=HERE/'clean-env'
    assert sys.prefix==str(environment),'Use the isolated environment executable'
    assert 'include-system-site-packages = false' in (environment/'pyvenv.cfg').read_text()
    for line in (ROOT/'requirements.lock').read_text().splitlines():
        if '==' in line:
            name,version=line.strip().split('==');assert importlib.metadata.version(name)==version
    manifest=json.loads((OLD/'manifest.json').read_text())
    rows=[]
    for task,dataset in [('classification','breast_cancer'),('regression','diabetes'),('clustering','iris')]:
        case=next(c for c in manifest['cases'] if c['task']==task and c['dataset']==dataset and c['seed']==42)
        out=HERE/'clean-smoke'/case['id'];out.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-B',str(ROOT/'main.py'),'--input',str(OLD/case['input']),
                 '--output',str(out),'--strategy','submission','--time-budget','50','--seed','42']
        started=time.monotonic()
        completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60,
                                 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        seconds=time.monotonic()-started
        (out/'stderr.txt').write_text(completed.stderr)
        assert completed.returncode==0,completed.stderr
        reference=json.loads((OLD/case['reference']).read_text())
        pred=pd.read_csv(out/'predictions.csv',dtype=str)
        assert list(pred)==['prediction'] and len(pred)==len(reference['truth']) and not pred.prediction.isna().any()
        if task=='classification':assert set(pred.prediction)<=set(reference['training_labels'])
        elif task=='regression':assert np.isfinite(pred.prediction.astype(float)).all()
        else:assert set(pred.prediction.astype(int))==set(range(case['k']))
        run=json.loads((out/'run.json').read_text())
        assert not run['model']['model'].startswith('Emergency')
        expected=HERE/'submission'/case['id']/'submission-0'
        expected_hash=hashlib.sha256((expected/'predictions.csv').read_bytes()).hexdigest()
        actual_hash=hashlib.sha256((out/'predictions.csv').read_bytes()).hexdigest()
        assert actual_hash==expected_hash,'STOP: clean-environment normal-size predictions changed'
        rows.append(dict(case_id=case['id'],seconds=seconds,valid=True,profile_fields=profile(case,out),
                         model=run['model']['model'],prediction_sha256=actual_hash,
                         normal_size_equivalent=True))
        print(json.dumps(rows[-1]),flush=True)
        check_freeze()
    (HERE/'clean-smoke.json').write_text(json.dumps(dict(environment=sys.prefix,packages_are_isolated=True,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
