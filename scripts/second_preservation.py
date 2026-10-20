"""Verify protected checkpoint files and unchanged estimator definitions; no fits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

CHECKPOINT = 'd770d8bbea25820dd0f4cbd231786e4427c7ca30'
ALLOWED = {f'src/unknown_dataset/{name}.py' for name in (
    'clustering', 'models', 'policies', 'routed', 'routing', 'runner', 'supervised')}


def git(*args):
    return subprocess.check_output(['git', *args])


def definition(source, name):
    return next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output; historical evidence is never overwritten')
    assert git('rev-parse', 'HEAD').decode().strip() == CHECKPOINT
    paths = git('ls-tree', '-r', '--name-only', CHECKPOINT).decode().splitlines()
    protected = []
    changed = []
    for path in paths:
        before = git('show', f'{CHECKPOINT}:{path}')
        current = Path(path).read_bytes()
        if before != current:
            assert path in ALLOWED, f'Unexpected tracked-file change: {path}'
            changed.append(path)
        else:
            protected.append(path)
    model_path = 'src/unknown_dataset/models.py'
    before_model = definition(git('show', f'{CHECKPOINT}:{model_path}'), 'fit_predict')
    after_model = definition(Path(model_path).read_bytes(), 'fit_predict')
    def baseline_body(node):
        start = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'started' for t in n.targets))
        return [ast.dump(n, include_attributes=False) for n in node.body[start:]]
    assert baseline_body(before_model) == baseline_body(after_model)
    supervised = 'src/unknown_dataset/supervised.py'
    before_fit = definition(git('show', f'{CHECKPOINT}:{supervised}'), '_fit')
    after_fit = definition(Path(supervised).read_bytes(), '_fit')
    assert ast.dump(before_fit) == ast.dump(after_fit)
    assert not git('diff', '--cached', '--name-only').strip(), 'Unexpected staged work'
    # Attributes deliberately preserve bytes; existing Windows CRLF is not
    # trailing source whitespace. Do not rewrite benchmarked source line endings.
    git('-c', 'core.whitespace=cr-at-eol', 'diff', '--check')
    record = dict(status='passed', checkpoint=CHECKPOINT,
                  unchanged_tracked_files=protected, intended_changed_tracked_files=changed,
                  baseline_estimator_body_unchanged=True,
                  supervised_estimator_definitions_unchanged=True,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path('src/unknown_dataset').glob('*.py'))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', unchanged_files=len(protected), changed=changed)))


if __name__ == '__main__':
    main()
