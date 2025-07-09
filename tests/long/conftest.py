import mlflow
import pytest


@pytest.fixture
def mlflow_run(tmp_path):
    tracking_uri = f'file://{tmp_path}/mlruns'
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run() as run:
        yield run


@pytest.fixture
def pyfunc_model():
    import mlflow.pyfunc

    class SimpleModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input

    return SimpleModel()


@pytest.fixture
def callable_model():
    def simple_callable(x):
        return x

    return simple_callable


@pytest.fixture
def catboost_model():
    import catboost
    import numpy as np

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = catboost.CatBoostClassifier(
        iterations=1,
        depth=1,
        learning_rate=1.0,
        verbose=False,
    )
    model.fit(X, y)

    return model


@pytest.fixture
def lightgbm_booster_model():
    import lightgbm as lgb
    import numpy as np

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'num_leaves': 2,
        'num_boost_round': 1,
        'verbose': -1,
    }

    model = lgb.train(params, train_data, num_boost_round=1)

    return model


@pytest.fixture
def lightgbm_lgbm_model():
    import lightgbm as lgb
    import numpy as np

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = lgb.LGBMClassifier(
        n_estimators=1,
        num_leaves=2,
        verbose=-1,
    )
    model.fit(X, y)

    return model


@pytest.fixture
def xgboost_booster_model():
    import numpy as np
    import xgboost as xgb

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'max_depth': 1,
        'eta': 1.0,
        'objective': 'binary:logistic',
        'verbosity': 0,
    }

    model = xgb.train(params, dtrain, num_boost_round=1)

    return model


@pytest.fixture
def xgboost_xgb_model():
    import numpy as np
    import xgboost as xgb

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = xgb.XGBClassifier(
        n_estimators=1,
        max_depth=1,
        learning_rate=1.0,
        verbosity=0,
    )
    model.fit(X, y)

    return model


@pytest.fixture
def sklearn_model():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = LogisticRegression(max_iter=1)
    model.fit(X, y)

    return model


@pytest.fixture
def torch_model():
    import torch
    from torch import nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[0.0], [1.0]])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    return model


@pytest.fixture
def tensorflow_model():
    import numpy as np
    import tensorflow as tf

    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    y = np.array([0, 1, 0], dtype=np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)),
    ])

    model.compile(optimizer='sgd', loss='binary_crossentropy')
    model.fit(X, y, epochs=1, verbose=0)

    return model


@pytest.fixture
def fastai_model():
    import pandas as pd
    from fastai.tabular.all import TabularDataLoaders

    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0],
    })

    dls = TabularDataLoaders.from_df(
        df,
        path='.',
        cat_names=[],
        cont_names=['feature1', 'feature2'],
        y_names='target',
        bs=2,
    )

    from fastai.tabular.model import tabular_learner

    learner = tabular_learner(dls, metrics=[])

    learner.fit_one_cycle(1, lr_max=0.1)

    return learner


@pytest.fixture
def mxnet_model():
    import mxnet as mx
    from mxnet.gluon import Block, nn

    class SimpleBlock(Block):
        def __init__(self):
            super().__init__()
            with self.name_scope():
                self.dense = nn.Dense(1)

        def forward(self, x):
            return self.dense(x)

    model = SimpleBlock()
    model.initialize(ctx=mx.cpu())

    x = mx.nd.array([[1.0, 2.0], [3.0, 4.0]])
    with mx.autograd.record():
        output = model(x)

    return model


@pytest.fixture
def statsmodels_model():
    import numpy as np
    import statsmodels.api as sm

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    X = sm.add_constant(X)

    model = sm.OLS(y, X)
    fitted_model = model.fit()

    return fitted_model


@pytest.fixture
def prophet_model():
    import pandas as pd
    from prophet import Prophet

    df = pd.DataFrame({
        'ds': pd.date_range('2021-01-01', periods=3),
        'y': [1, 2, 3],
    })

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
    )
    model.fit(df)

    return model


@pytest.fixture
def paddlepaddle_model():
    import paddle
    from paddle import nn

    class SimpleModel(nn.Layer):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    y = paddle.to_tensor([[0.0], [1.0]])

    optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())
    loss_fn = nn.MSELoss()

    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    return model


@pytest.fixture
def spacy_model():
    import spacy

    nlp = spacy.blank('en')

    nlp.add_pipe('tagger')

    return nlp


@pytest.fixture
def h2o_model():
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator

    h2o.init()

    data = [[1, 2, 0], [3, 4, 1], [5, 6, 0]]
    df = h2o.H2OFrame(data)
    df.set_names(['feature1', 'feature2', 'target'])

    model = H2OGradientBoostingEstimator(
        ntrees=1,
        max_depth=1,
        learn_rate=1.0,
    )
    model.train(x=['feature1', 'feature2'], y='target', training_frame=df)

    return model


@pytest.fixture
def pyfunc_model_and_signature(pyfunc_model):
    input_example = {'col1': [1, 2, 3]}
    output_example = {'col1': [1, 2, 3]}
    return pyfunc_model, input_example, output_example


@pytest.fixture
def callable_model_and_signature(callable_model):
    input_example = [1, 2, 3]
    output_example = [1, 2, 3]
    return callable_model, input_example, output_example


@pytest.fixture
def catboost_model_and_signature(catboost_model):
    input_example = [[1, 2], [3, 4], [5, 6]]
    output_example = [0, 1, 0]
    return catboost_model, input_example, output_example


@pytest.fixture
def lightgbm_booster_model_and_signature(lightgbm_booster_model):
    input_example = [[1, 2], [3, 4], [5, 6]]
    output_example = [0, 1, 0]
    return lightgbm_booster_model, input_example, output_example


@pytest.fixture
def lightgbm_lgbm_model_and_signature(lightgbm_lgbm_model):
    input_example = [[1, 2], [3, 4], [5, 6]]
    output_example = [0, 1, 0]
    return lightgbm_lgbm_model, input_example, output_example


@pytest.fixture
def xgboost_booster_model_and_signature(xgboost_booster_model):
    input_example = [[1, 2], [3, 4], [5, 6]]
    output_example = [0, 1, 0]
    return xgboost_booster_model, input_example, output_example


@pytest.fixture
def xgboost_xgb_model_and_signature(xgboost_xgb_model):
    input_example = [[1, 2], [3, 4], [5, 6]]
    output_example = [0, 1, 0]
    return xgboost_xgb_model, input_example, output_example


@pytest.fixture
def sklearn_model_and_signature(sklearn_model):
    input_example = [[1, 2], [3, 4], [5, 6]]
    output_example = [0, 1, 0]
    return sklearn_model, input_example, output_example


@pytest.fixture
def torch_model_and_signature(torch_model):
    input_example = [[1.0, 2.0], [3.0, 4.0]]
    output_example = [[0.0], [1.0]]
    return torch_model, input_example, output_example


@pytest.fixture
def tensorflow_model_and_signature(tensorflow_model):
    input_example = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    output_example = [0, 1, 0]
    return tensorflow_model, input_example, output_example


@pytest.fixture
def fastai_model_and_signature(fastai_model):
    input_example = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    output_example = [0, 1, 0]
    return fastai_model, input_example, output_example


@pytest.fixture
def mxnet_model_and_signature(mxnet_model):
    input_example = [[1.0, 2.0], [3.0, 4.0]]
    output_example = [[0.0], [1.0]]
    return mxnet_model, input_example, output_example


@pytest.fixture
def statsmodels_model_and_signature(statsmodels_model):
    input_example = [[1, 2], [3, 4], [5, 6]]
    output_example = [0, 1, 0]
    return statsmodels_model, input_example, output_example


@pytest.fixture
def prophet_model_and_signature(prophet_model):
    input_example = {'ds': ['2021-01-01', '2021-01-02', '2021-01-03']}
    output_example = [1, 2, 3]
    return prophet_model, input_example, output_example


@pytest.fixture
def paddlepaddle_model_and_signature(paddlepaddle_model):
    input_example = [[1.0, 2.0], [3.0, 4.0]]
    output_example = [[0.0], [1.0]]
    return paddlepaddle_model, input_example, output_example


@pytest.fixture
def spacy_model_and_signature(spacy_model):
    input_example = ['This is a sentence.']
    output_example = [['This', 'is', 'a', 'sentence', '.']]
    return spacy_model, input_example, output_example


@pytest.fixture
def h2o_model_and_signature(h2o_model):
    input_example = [[1, 2], [3, 4], [5, 6]]
    output_example = [0, 1, 0]
    return h2o_model, input_example, output_example
