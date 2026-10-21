"""Serial diagnosis of the nine authorized historical failure combinations."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/real-data-check'


def main():
    manifest = json.loads((OLD / 'manifest.json').read_text())
    plans = []
    for task, shape, strategies in [
        ('classification', '100000x50', ['baseline', 'routed']),
        ('regression', '100000x50', ['baseline', 'auto', 'routed']),
        ('regression', '5000x500', ['baseline', 'auto', 'routed']),
        ('classification', '5000x500', ['auto']),
    ]:
        case = next(c for c in manifest['cases'] if
                    c['id'] == f'stress-{task}-shape_{shape}-7001')
        plans.extend((case, strategy) for strategy in strategies)
    record = HERE / 'diagnosis.jsonl'
    done = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
    for case, strategy in plans:
        if any(r['case_id'] == case['id'] and r['strategy'] == strategy for r in done):
            continue
        output = HERE / 'diagnosis' / case['id'] / strategy
        output.mkdir(parents=True, exist_ok=False)
        for name, expected in case['input_sha256'].items():
            assert hashlib.sha256((OLD / case['input'] / name).read_bytes()).hexdigest() == expected
        command = [sys.executable, '-B', str(ROOT / 'main.py'), '--input',
                   str(OLD / case['input']), '--output', str(output), '--strategy', strategy,
                   '--seed', str(case['seed']), '--time-budget', '50', '--timings']
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(HERE / 'trace'), RELIABILITY_TRACE_DIR=str(output))
        started = time.monotonic()
        with (output / 'stdout.txt').open('w') as out, (output / 'stderr.txt').open('w') as err:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
            timeout = False
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                timeout = True
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                process.wait(timeout=10)
        row = dict(case_id=case['id'], strategy=strategy, seconds=time.monotonic()-started,
                   returncode=process.returncode, timeout=timeout,
                   stderr=(output / 'stderr.txt').read_text(),
                   run=json.loads((output / 'run.json').read_text()) if (output / 'run.json').exists() else None)
        with record.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps({k: row[k] for k in ['case_id', 'strategy', 'seconds', 'returncode']}), flush=True)


if __name__ == '__main__':
    main()
