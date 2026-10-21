"""Run authorized verification phases serially and preserve brief phase notes."""
from pathlib import Path
import datetime
import subprocess
import sys

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    for phase in ['stress','real','submission']:
        subprocess.run([sys.executable,'-B',str(HERE/'verify.py'),phase],check=True)
        with (HERE/'PROGRESS.md').open('a') as h:
            h.write(f'\nPhase 5 {phase} CLI verification completed successfully at '
                    f'{datetime.datetime.now(datetime.timezone.utc).isoformat()}. '
                    f'Per-run evidence: {phase}.jsonl.\n')
        print('PHASE COMPLETE '+phase,flush=True)
