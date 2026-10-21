"""Record and archive a validated source state without creating a Git commit."""
from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

from verify import HERE, ROOT, OLD, source_hashes


def main():
    path=HERE/'freeze.json'
    assert not path.exists(),'Freeze already exists; do not replace it'
    now=datetime.datetime.now(datetime.timezone.utc)
    cutoff=datetime.datetime(2026,10,21,12,tzinfo=datetime.timezone.utc)
    assert now<cutoff,'Eight-hour freeze cutoff missed: STOP'
    for name,count in [('nine',9),('stress',39),('real',108),('submission',49)]:
        rows=[json.loads(l) for l in (HERE/f'{name}.jsonl').read_text().splitlines()]
        assert len(rows)==count and all(r['valid'] and not r.get('error') for r in rows),name
        assert all(not r['model']['model'].startswith('Emergency') for r in rows),name
        assert max(r['seconds'] for r in rows)<60,name
        if name=='real':assert all(r.get('historical_equivalent') for r in rows)
    audit=json.loads((HERE/'profile-audit.json').read_text())
    assert len(set(r['case_id'] for r in audit))==13
    suite=ET.parse(HERE/'full-tests.xml').getroot()
    nodes=[n for n in suite.iter('testsuite')]
    assert sum(int(n.get('tests',0)) for n in nodes)==295
    assert not any(int(n.get('errors',0))+int(n.get('failures',0)) for n in nodes)
    hashes=source_hashes()
    canonical=json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()
    archive=HERE/'frozen-source.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in hashes:z.write(ROOT/name,name)
    versions={}
    for line in (ROOT/'requirements.lock').read_text().splitlines():
        if '==' in line:
            name,expected=line.strip().split('==')
            versions[name]=importlib.metadata.version(name)
            assert versions[name]==expected,(name,versions[name],expected)
    manifest=json.loads((OLD/'manifest.json').read_text())
    record=dict(frozen_at_utc=now.isoformat(),deadline_utc='2026-10-21T20:00:00+00:00',
                cutoff_utc=cutoff.isoformat(),checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_sha256=hashes,source_manifest_sha256=hashlib.sha256(canonical).hexdigest(),
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                environment=dict(python=sys.version,platform=platform.platform(),packages=versions,
                                 numerical_threads='main.py enforces one thread before numerical imports'),
                command='python main.py --input INPUT --output OUTPUT --strategy submission --time-budget 50 --seed SEED',
                input_manifest_sha256=hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),
                input_hashes={c['id']:dict(files=c['input_sha256'],reference_sha256=c['reference_sha256']) for c in manifest['cases']},
                status='frozen; final verification pending; no post-freeze source/config/README/dependency edits permitted')
    path.write_text(json.dumps(record,indent=2)+'\n')
    with (HERE/'PROGRESS.md').open('a') as h:
        h.write(f'\nPhase 6 source freeze recorded at {now.isoformat()}; source manifest '
                f'{record["source_manifest_sha256"]}. Do not edit frozen files. '
                'Final submission verification and isolated-environment smoke remain pending.\n')
    print(json.dumps({k:record[k] for k in ['frozen_at_utc','source_manifest_sha256','archive_sha256','command']},indent=2))


if __name__=='__main__':main()
