"""Summarize pairwise agreement of newly generated solver coefficient vectors."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))


def summarize(output, config):
    import numpy as np
    import designs
    from pair_metrics import pair_metrics, METRICS
    from threadpoolctl import threadpool_limits

    output = Path(output)
    destination = output/'summary'
    destination.mkdir(exist_ok=True)
    cells = {cell['id']: cell for cell in designs.cells(config)}
    inspected = {(cell_id, config['run_split'], draw) for cell_id, draw in config['inspected_datasets']}
    groups = {}
    pair_rows = datasets = excluded_completed = 0
    # An ephemeral index keeps full coefficient vectors on disk rather than
    # loading the complete 480,000-fit record collection into Python memory.
    with tempfile.TemporaryDirectory(prefix='bbp-agreement-') as temporary:
        db = sqlite3.connect(Path(temporary)/'roots.sqlite')
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('''CREATE TABLE fits(cell_id TEXT,split TEXT,draw INTEGER,equation TEXT,
            method TEXT,beta BLOB,strict_raw INTEGER,loose_raw INTEGER,strict_cap INTEGER,
            loose_cap INTEGER,PRIMARY KEY(cell_id,split,draw,equation,method))''')
        with (output/'fits.jsonl').open() as stream, db:
            for line in stream:
                row = json.loads(line)
                beta = row['beta']
                blob = None if beta is None else np.asarray(beta,dtype='<f8').tobytes()
                db.execute('INSERT INTO fits VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (row['cell_id'],row['split'],row['draw'],row['estimator'],row['solver'],blob,
                     int(row['accepted_1e8']),int(row['accepted_1e6']),
                     int(row['common_cap_accepted_1e8']),int(row['common_cap_accepted_1e6'])))
        keys = db.execute('SELECT DISTINCT cell_id,split,draw FROM fits ORDER BY cell_id,split,draw')
        with threadpool_limits(limits=1):
            for cell_id, split, draw in keys:
                key = (cell_id,split,draw)
                data = designs.generate(cells[cell_id],draw,split,config['master_seed'])
                records = db.execute('SELECT equation,method,beta,strict_raw,loose_raw,strict_cap,loose_cap '
                    'FROM fits WHERE cell_id=? AND split=? AND draw=? ORDER BY equation,method',key).fetchall()
                equations = {}
                for equation,method,blob,*flags in records:
                    equations.setdefault(equation,{})[method] = (
                        np.frombuffer(blob,dtype='<f8') if blob is not None else None,flags)
                for equation, methods in equations.items():
                    for method_a, method_b in itertools.combinations(sorted(methods),2):
                        beta_a, flags_a = methods[method_a]
                        beta_b, flags_b = methods[method_b]
                        metrics,_ = pair_metrics(data['X'],beta_a,beta_b,designs.TREATMENT_COLUMN)
                        pair_rows += 1
                        for selection in ['full','exclude_inspected20']:
                            if selection == 'exclude_inspected20' and key in inspected:
                                continue
                            for view, offset in [('raw',0),('common_cap',2)]:
                                for gate, gate_offset in [('strict',0),('loose',1)]:
                                    group_key=(selection,view,gate,cell_id,equation,method_a,method_b)
                                    entry = groups.setdefault(group_key,dict(
                                        selection=selection,view=view,gate=gate,cell_id=cell_id,
                                        estimator=equation,solver_a=method_a,solver_b=method_b,
                                        available_dataset_pairs=0,common_accepted_pairs=0,finite_vector_pairs=0))
                                    entry['available_dataset_pairs'] += 1
                                    entry['finite_vector_pairs'] += metrics['finite_pair']
                                    if not (flags_a[offset+gate_offset] and flags_b[offset+gate_offset]):
                                        continue
                                    entry['common_accepted_pairs'] += 1
                                    for metric in METRICS:
                                        value=metrics[metric]
                                        if value is not None:
                                            count,maximum=metric+'_finite_count',metric+'_maximum'
                                            entry[count]=entry.get(count,0)+1
                                            entry[maximum]=max(value,entry.get(maximum,value))
                datasets += 1
                excluded_completed += int(key in inspected)
        db.close()
    fields=['selection','view','gate','cell_id','estimator','solver_a','solver_b',
            'available_dataset_pairs','common_accepted_pairs','finite_vector_pairs']
    for metric in METRICS:
        fields += [metric+'_finite_count',metric+'_maximum']
    with (destination/'agreement_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for _,entry in sorted(groups.items()):
            for metric in METRICS:
                entry.setdefault(metric+'_finite_count',0)
                entry.setdefault(metric+'_maximum',None)
            writer.writerow(entry)
    counts=dict(datasets_completed=datasets,expected_datasets=len(cells)*config['draws_per_cell'],
        pair_rows=pair_rows,expected_pair_rows=len(cells)*config['draws_per_cell']*30,
        inspected_exclusion_keys=len(inspected),inspected_keys_among_completed=excluded_completed,
        pairs_after_inspected_exclusion=sum(entry['available_dataset_pairs'] for key,entry in groups.items()
            if key[0]=='exclude_inspected20' and key[1]=='raw' and key[2]=='strict'),
        metrics_note='Within-equation distances only; no new agreement tolerance or cross-equation equality.')
    (destination/'agreement_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    return destination


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((ROOT/'config.json').read_text())
    print(summarize(args.output,config))
