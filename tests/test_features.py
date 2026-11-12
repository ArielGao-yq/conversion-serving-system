import numpy as np
import pandas as pd 
from src.features import build_preprocessor

NUMERIC=['age']
CATEGORICAL=['job']

def make_train():
    return pd.DataFrame(
        {'age':[25,35,35],
        'job':['admin','technician','admin'],
        }
    )

def test_output_is_dense_with_no_nans():
    pre = build_preprocessor(NUMERIC, CATEGORICAL)
    out = pre.fit_transform(make_train())
    assert isinstance(out, np.ndarray)
    assert out.shape==(3,3)
    assert not np.isnan(out).any()

def test_numeric_column_is_scaled():
    pre=build_preprocessor(NUMERIC, CATEGORICAL)
    out=pre.fit_transform(make_train())
    assert abs(out[:,0].mean())<1e-9

def test_unseen_category_does_not_crash():
    pre=build_preprocessor(NUMERIC, CATEGORICAL)
    out=pre.fit_transform(make_train())
    new = pd.DataFrame({"age": [30], "job": ["astronaut"]})
    out = pre.transform(new)
    assert out.shape == (1, 3)
    assert not np.isnan(out).any()
