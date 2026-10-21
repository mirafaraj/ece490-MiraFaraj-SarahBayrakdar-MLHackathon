"""Summarize completed and interrupted call boundaries without inventing timings."""
from pathlib import Path
import collections
import json

HERE = Path(__file__).resolve().parent


def summarize():
    rows = []
    for line in (HERE / 'diagnosis.jsonl').read_text().splitlines():
        row = json.loads(line)
        directory = HERE / 'diagnosis' / row['case_id'] / row['strategy']
        processes = []
        for path in sorted(directory.glob('events-*.tsv')):
            stacks = collections.defaultdict(list)
            totals = collections.defaultdict(float)
            counts = collections.Counter()
            start = None
            last = None
            for text in path.read_text().splitlines():
                parts = text.split('\t')
                stamp, event, tag = float(parts[0]), parts[1], parts[2]
                if start is None:
                    start = stamp
                last = stamp
                if event == 'call':
                    stacks[tag].append(stamp)
                if event == 'return' and stacks[tag]:
                    totals[tag] += stamp - stacks[tag].pop()
                    counts[tag] += 1
            processes.append(dict(pid=path.stem, start=start, last_event=last,
                                  completed_seconds=dict(totals), completed_calls=dict(counts),
                                  active_calls={k:v for k,v in stacks.items() if v}))
        rows.append(dict(case_id=row['case_id'], strategy=row['strategy'],
                         seconds=row['seconds'], returncode=row['returncode'],
                         timeout=row['timeout'], processes=processes,
                         parent_phases=row['run'].get('phase_seconds') if row['run'] else None,
                         worker_phases=row['run'].get('worker_phase_seconds') if row['run'] else None,
                         final_model=row['run']['model'] if row['run'] else None))
    (HERE / 'diagnosis-summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    for row in rows:
        print(row['case_id'], row['strategy'], round(row['seconds'],3),row['returncode'])
        for process in row['processes']:
            print(' ',process['pid'], 'completed',
                  {k:round(v,3) for k,v in process['completed_seconds'].items()
                   if any(t in k for t in [':fit:', '_fit:', 'validate_predictions', 'build_profile', 'write_outputs'])},
                  'ACTIVE',process['active_calls'])


if __name__ == '__main__':
    summarize()
