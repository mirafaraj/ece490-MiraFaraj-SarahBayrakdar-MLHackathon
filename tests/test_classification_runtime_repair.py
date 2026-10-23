"""Fixed full-training admission, preserving ordinary and numeric-only paths."""
from types import SimpleNamespace
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from unknown_dataset import reliability


def test_prepared_width_accounts_for_missingness_and_one_hot_limits():
    from unknown_dataset.preprocessing import TableCleaner, build_preprocessor
    frame = pd.DataFrame({'x': [1., np.nan, 2., 3.] * 30,
                          'c': [str(i) for i in range(120)],
                          'd': ['a', None, 'b', 'a'] * 30})
    kinds = {'x': 'numeric', 'c': 'categorical', 'd': 'categorical'}
    assert reliability.prepared_mixed_width(frame, kinds) == 2 + 32 + 3
    cleaner = TableCleaner(kinds).fit(frame)
    encoded = build_preprocessor(cleaner).fit_transform(cleaner.transform(frame))
    assert encoded.shape[1] == reliability.prepared_mixed_width(frame, kinds)
    # Constants and duplicate columns can overestimate, never remove needed width.
    frame['duplicate'] = frame.c
    frame['constant'] = 'fixed'
    assert reliability.prepared_mixed_width(frame) >= encoded.shape[1]


def test_numeric_cleaning_and_lexical_categories_match_width_bound():
    frame = pd.DataFrame({'x': ['1', 'bad', 'inf', '2'],
                          'c': pd.Series([1, True, '1', None], dtype=object)})
    assert reliability.prepared_mixed_width(frame, {'x':'numeric', 'c':'categorical'}) == 5
    assert reliability.prepared_mixed_width(pd.DataFrame({'x':[1.,np.nan]})) is None


@pytest.mark.parametrize('strategy', ['baseline','auto','submission'])
def test_fixed_row_and_full_matrix_boundaries(monkeypatch, strategy):
    frame = pd.DataFrame({'category':['a','b']})
    monkeypatch.setattr(reliability, 'prepared_mixed_width', lambda *args: 80)
    check = lambda rows: reliability.needs_bounded_forest('classification', rows, 1, strategy, frame)
    assert not check(20_000)
    assert not check(25_000)  # Exactly 2,000,000 prepared cells is allowed.
    assert check(25_001)
    monkeypatch.setattr(reliability, 'prepared_mixed_width', lambda *args: 101)
    assert not check(20_000)
    assert check(20_001)


def test_other_tasks_and_existing_numeric_paths_do_not_gain_admission():
    numeric = pd.DataFrame({'x':[1.,2.,3.]})
    assert not reliability.needs_bounded_forest('classification',100_000,50,'auto',numeric)
    mixed = numeric.assign(category=['a','b','c'])
    assert not reliability.needs_bounded_forest('regression',100_000,2,'auto',mixed)
    assert not reliability.needs_bounded_forest('clustering',100_000,2,'auto',mixed)
    assert not reliability.needs_bounded_forest('classification',100_000,2,'routed',mixed)


@pytest.mark.parametrize('strategy',['baseline','auto','submission'])
def test_public_api_admits_before_unbounded_selection_or_fit(monkeypatch,strategy):
    from unknown_dataset import models, supervised
    frame = pd.DataFrame({'x':[1.,2.], 'category':['a','b']})
    observed=[]
    def admission(task,rows,features,chosen,X_train,feature_types):
        observed.append((X_train is frame, chosen));return True
    monkeypatch.setattr(reliability,'needs_bounded_forest',admission)
    monkeypatch.setattr(reliability,'fit_predict_bounded',lambda *args: (np.array(['a']),{'bounded':True}))
    monkeypatch.setattr(supervised,'fit_predict_supervised',lambda **kwargs:pytest.fail('Unbounded selector entered'))
    prediction,info=models.fit_predict('classification',frame,['a','b'],frame.iloc[:1],strategy=strategy)
    assert info['bounded'] and prediction.tolist()==['a']
    assert observed==[(True, 'auto' if strategy=='submission' else strategy)]


def test_classification_forest_failure_reuses_linear_with_same_absolute_deadline(monkeypatch,tmp_path):
    from unknown_dataset import runner
    attempts=[]
    data=SimpleNamespace(task='classification',X_train=pd.DataFrame({'x':[1,2,3]}),
                         y_train=pd.Series(['a','b','a']),X_test=pd.DataFrame({'x':[4]}))
    class Process:
        exitcode=0
        def __init__(self,target,args):self.args=args
        def start(self):
            mode=self.args[-1];attempts.append((mode,self.args[-2]))
            payload={'error':'forced forest failure'} if mode=='forest' else {
                'predictions':np.array(['a']),'model_info':{'model':'LogisticRegression','fallback_kind':'bounded_linear'}}
            with open(self.args[3],'wb') as handle:pickle.dump(payload,handle)
        def join(self,*args):pass
        def is_alive(self):return False
    monkeypatch.setattr(runner.mp,'get_context',lambda _:SimpleNamespace(Process=Process))
    deadline=time.monotonic()+20
    result=runner._bounded_worker_result(data,47131,'auto',None,str(tmp_path),deadline,True)
    assert attempts==[('forest',deadline),('linear',deadline)]
    assert result['model_info']['model']=='LogisticRegression'
    assert [r['outcome'] for r in result['model_info']['reliability_attempts']]==['failed','completed']
