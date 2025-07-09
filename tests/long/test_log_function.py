import pytest
from pytest_lazyfixture import lazy_fixture as lf

from mlflow_polymodel import log_model


@pytest.mark.parametrize(
    ('model'),
    [
        pytest.param(lf('function_model')),
        pytest.param(lf('pyfunc_model')),
        pytest.param(lf('catboost_model'), marks=pytest.mark.slow),
        pytest.param(lf('lightgbm_booster_model'), marks=pytest.mark.slow),
        pytest.param(lf('lightgbm_lgbm_model'), marks=pytest.mark.slow),
        pytest.param(lf('xgboost_booster_model'), marks=pytest.mark.slow),
        pytest.param(lf('xgboost_xgb_model'), marks=pytest.mark.slow),
        pytest.param(lf('sklearn_model'), marks=pytest.mark.slow),
        pytest.param(lf('torch_model'), marks=pytest.mark.slow),
        pytest.param(lf('tensorflow_model'), marks=pytest.mark.slow),
        pytest.param(lf('fastai_model'), marks=pytest.mark.slow),
        pytest.param(lf('mxnet_model'), marks=pytest.mark.slow),
        pytest.param(lf('statsmodels_model'), marks=pytest.mark.slow),
        pytest.param(lf('prophet_model'), marks=pytest.mark.slow),
        pytest.param(lf('paddlepaddle_model'), marks=pytest.mark.slow),
        pytest.param(lf('spacy_model'), marks=pytest.mark.slow),
        pytest.param(lf('h2o_model'), marks=pytest.mark.slow),
    ],
)
def test_log(model, tmp_path):
    log_model(model, artifact_path=tmp_path)
    assert_model_logged(tmp_path)
