"""Fixed resource guards, strict label membership and learned recovery contracts."""
from pathlib import Path
from types import SimpleNamespace
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from unknown_dataset.checkpoints import validate_predictions
from unknown_dataset.reliability import (effective_strategy, needs_bounded_forest,
    guard_histogram, fitting_indices, fit_predict_bounded)


def test_resource_rules_preserve_successful_alternatives():
    for strategy in ['baseline','auto','routed','submission']:
        assert not needs_bounded_forest('regression',15_480,8,strategy)
        assert not needs_bounded_forest('classification',8387,6,strategy)
        assert not needs_bounded_forest('clustering',100_000,50,strategy)
        assert needs_bounded_forest('regression',5000,500,strategy)
    assert needs_bounded_forest('classification',100_000,50,'baseline')
    assert needs_bounded_forest('classification',100_000,50,'routed')
    assert not needs_bounded_forest('classification',100_000,50,'auto')
    assert not needs_bounded_forest('classification',5000,500,'baseline')
    assert guard_histogram(5000,500)
    assert not guard_histogram(100_000,50)
    assert not guard_histogram(1095,80)


def test_bounded_sampling_preserves_singletons_and_is_training_only():
    labels=np.array(['common']*21000+['rare']*1500+['singleton'])
    indices=fitting_indices(labels,'classification',711)
    assert len(indices)==20_000 and len(set(indices))==20_000
    assert set(labels[indices])==set(labels)
    assert np.array_equal(indices,fitting_indices(labels,'classification',711))
    assert np.all(indices[:-1]<indices[1:])
    assert 1300 < np.sum(labels[indices]=='rare') < 1400
    regression=fitting_indices(np.arange(30000),'regression',17)
    assert len(regression)==20_000 and len(set(regression))==20_000
    assert np.array_equal(fitting_indices(['a','b'],'classification',1),[0,1])
    with pytest.raises(ValueError,match='preserve all classes'):
        fitting_indices(np.arange(20_001),'classification',1)


@pytest.mark.parametrize('labels',[
    np.array(['01','1','001','1.0','1e0']*9),
    np.array([1,'1',1.0,True],dtype=object),
    np.array(['a','b']*12000),
])
@pytest.mark.parametrize('prediction',[
    ['01','1'],['001','1.0'],['a','b'],[1,'1'],[True,1.0],['missing','a'],[None,'1'],[np.inf,'1'],
])
def test_membership_matches_previous_predicate_exactly(labels,prediction):
    values=np.asarray(prediction,dtype=object)
    old=(not pd.isna(values).any() and np.isin(values,np.asarray(labels)).all()
         and not any(isinstance(v,(float,np.floating)) and not np.isfinite(v) for v in values))
    data=SimpleNamespace(task='classification',X_test=[0,1],y_train=labels)
    if old:
        np.testing.assert_array_equal(validate_predictions(data,prediction),values)
    else:
        with pytest.raises(ValueError):validate_predictions(data,prediction)


def test_fixed_forest_parameters_are_isolated(monkeypatch):
    from unknown_dataset import models
    seen=[]
    class Estimator:
        def __init__(self,**kwargs):seen.append(kwargs)
        def fit(self,x,y):return self
        def predict(self,x):return np.zeros(len(x))
    monkeypatch.setattr(models,'ExtraTreesRegressor',Estimator)
    frame=pd.DataFrame({'x':np.arange(40,dtype=float)})
    _,info=models.fit_predict('regression',frame,np.arange(40),frame.iloc[:2],strategy='baseline',_bounded_forest=True)
    assert seen[-1]['n_estimators']==16 and seen[-1]['max_features']=='sqrt'
    models.fit_predict('regression',frame,np.arange(40),frame.iloc[:2],strategy='baseline')
    assert seen[-1]['n_estimators']==96 and 'max_features' not in seen[-1]


def test_submission_delegates_by_task_and_is_api_default(monkeypatch):
    from unknown_dataset import models,supervised,clustering
    import inspect
    assert inspect.signature(models.fit_predict).parameters['strategy'].default=='submission'
    assert effective_strategy('classification','submission')=='baseline'
    assert effective_strategy('regression','submission')=='auto'
    assert effective_strategy('clustering','submission')=='auto'
    calls=[]
    def candidate(**kwargs):calls.append(kwargs['task']);return np.array([0]),{}
    monkeypatch.setattr(supervised,'fit_predict_supervised',candidate)
    monkeypatch.setattr(clustering,'fit_predict_clustering',candidate)
    frame=pd.DataFrame({'x':[1,2,3]})
    models.fit_predict('regression',frame,[1,2,3],frame.iloc[:1],strategy='submission')
    models.fit_predict('clustering',frame,n_clusters=1,strategy='submission')
    models.fit_predict('regression',frame,[1,2,3],frame.iloc[:1])
    models.fit_predict('clustering',frame,n_clusters=1)
    assert calls==['regression','clustering','regression','clustering']


@pytest.mark.parametrize('forest_outcome',['error','timeout','invalid'])
def test_forest_failure_uses_existing_linear_with_same_deadline(monkeypatch,tmp_path,forest_outcome):
    from unknown_dataset import runner
    modes=[]
    data=SimpleNamespace(task='regression',X_train=pd.DataFrame({'x':[1,2,3]}),
                         y_train=pd.Series([1.,2.,3.]),X_test=pd.DataFrame({'x':[4.]}))
    class Process:
        exitcode=0
        def __init__(self,target,args):self.args=args;self.alive=False
        def start(self):
            mode=self.args[-1];modes.append(mode)
            self.alive=mode=='forest' and forest_outcome=='timeout'
            payload={'error':'deliberate failure'} if mode=='forest' else {
                'predictions':np.array([4.]),'model_info':{'model':'Ridge','fallback_kind':'bounded_linear'}}
            if mode=='forest' and forest_outcome=='invalid':
                payload={'predictions':np.array([float('nan')]),'model_info':{'model':'invalid'}}
            if not self.alive:
                with open(self.args[3],'wb') as h:pickle.dump(payload,h)
        def join(self,*args):pass
        def is_alive(self):return self.alive
        def terminate(self):self.alive=False
    monkeypatch.setattr(runner.mp,'get_context',lambda _:SimpleNamespace(Process=Process))
    deadline=time.monotonic()+20
    result=runner._bounded_worker_result(data,42,'baseline',None,str(tmp_path),deadline,False)
    assert modes==['forest','linear'] and result['model_info']['model']=='Ridge'
    assert result['model_info']['reliability_attempts'][0]['outcome']=='failed'
    assert result['model_info']['reliability_attempts'][1]['outcome']=='completed'


def test_bounded_linear_is_learned_not_dummy():
    frame=pd.DataFrame({'x':np.arange(80,dtype=float)})
    prediction,info=fit_predict_bounded('regression',frame,frame.x*3+7,frame.iloc[-5:],
                                       {'x':'numeric'},42,10,mode='linear')
    assert info['fallback_kind']=='bounded_linear' and info['model']=='Ridge'
    assert np.mean(np.abs(prediction-(frame.x.iloc[-5:]*3+7)))<2
