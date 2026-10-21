"""Independent full-CSV reconstruction of the repository's interpreted schema.

Does not import production loading or profiling. Numeric calculations use Python
math/statistics; duplicate detection uses exact full-column tokens, not pandas hashes.
"""
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import pandas as pd


def numeric_summary(series):
    values = [float(x) for x in series if pd.notna(x) and math.isfinite(float(x))]
    if not values:
        return dict.fromkeys(['min', 'max', 'mean', 'std', 'median'])
    return dict(min=min(values), max=max(values), mean=math.fsum(values)/len(values),
                std=math.sqrt(statistics.pvariance(values)), median=statistics.median(values))


def reconstruct(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text())
    declared = meta.get('feature_types', {})
    lexical = [k for k,v in declared.items() if v == 'categorical']
    if meta['task'] == 'classification':
        lexical.append(meta['target'])
    table = pd.read_csv(directory / 'train.csv', dtype={k:str for k in lexical})
    target = table.pop(meta['target']) if meta['task'] != 'clustering' else None
    if meta.get('id_column'):
        table.pop(meta['id_column'])
    kinds = declared or {k:'numeric' if pd.api.types.is_numeric_dtype(table[k])
                         else 'categorical' for k in table}
    for k in table:
        if kinds[k] == 'numeric':
            table[k] = pd.to_numeric(table[k], errors='raise')
        else:
            table[k] = table[k].map(lambda x: str(x) if pd.notna(x) else float('nan'))
    columns, constants, duplicates, seen = {}, [], [], {}
    for k in table:
        series = table[k]
        missing = sum(pd.isna(x) for x in series)
        unique = len(set(x for x in series if pd.notna(x)))
        constant = unique <= 1
        if constant:
            constants.append(k)
        columns[k] = dict(type=kinds[k], missing_count=missing,
                          missing_fraction=missing/len(table), unique_count=unique,
                          constant=constant, all_missing=missing == len(table))
        if kinds[k] == 'numeric':
            columns[k]['numeric_summary'] = numeric_summary(series)
        # Include dtype and exact missing positions, like the stated schema.
        tokens = (str(series.dtype), tuple(None if pd.isna(x) else x for x in series))
        if tokens in seen:
            duplicates.append(dict(column=k, duplicate_of=seen[tokens]))
        else:
            seen[tokens] = k
    result = dict(schema_version='provisional-1', row_count=len(table), feature_count=len(table.columns),
                  feature_type_counts=dict(Counter({'numeric':0, 'categorical':0}) + Counter(kinds.values())),
                  missing_count=sum(v['missing_count'] for v in columns.values()),
                  constant_definition='At most one distinct nonmissing value; includes all-missing columns.',
                  constant_columns=constants, duplicate_columns=duplicates, columns=columns)
    result['feature_type_counts'] = {kind:list(kinds.values()).count(kind) for kind in ['numeric','categorical']}
    if target is not None:
        if meta['task'] == 'regression':
            target = pd.to_numeric(target, errors='raise')
        numeric = pd.api.types.is_numeric_dtype(target)
        counts = Counter(x for x in target if pd.notna(x))
        result['target'] = dict(name=target.name, missing_count=sum(pd.isna(x) for x in target),
                                unique_count=len(counts))
        if numeric:
            result['target']['numeric_summary'] = numeric_summary(target)
        if not numeric or len(counts) <= 50:
            result['target']['distribution'] = [dict(value=x, count=n) for x,n in counts.items()]
    return result


def compare(expected, actual, path='profile'):
    """All discrete fields exact; floats rtol=1e-10 and atol=1e-12."""
    if isinstance(expected, dict):
        assert isinstance(actual,dict) and expected.keys() == actual.keys(), path
        return sum(compare(v,actual[k],path+'/'+k) for k,v in expected.items())
    if isinstance(expected, list):
        assert isinstance(actual,list) and len(expected) == len(actual), path
        return sum(compare(v,actual[i],path+'/'+str(i)) for i,v in enumerate(expected))
    if isinstance(expected,float):
        assert isinstance(actual,(int,float)) and math.isclose(expected,actual,rel_tol=1e-10,abs_tol=1e-12), (path,expected,actual)
    else:
        assert expected == actual and type(expected) is type(actual), (path,expected,actual)
    return 1
