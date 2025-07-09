import mlflow
import pytest
from pytest_lazy_fixtures import lf

from mlflow_polymodel import log_model


def assert_valide_model_output(output, output_example) -> bool:
    print(output, output_example)
    return True


@pytest.mark.parametrize(
    ('model_and_signature'),
    [
        pytest.param(lf('callable_model_and_signature')),
        pytest.param(lf('pyfunc_model_and_signature')),
        pytest.param(lf('catboost_model_and_signature'), marks=pytest.mark.slow),
        pytest.param(lf('lightgbm_booster_model_and_signature'), marks=pytest.mark.slow),
        pytest.param(lf('lightgbm_lgbm_model_and_signature'), marks=pytest.mark.slow),
        pytest.param(lf('xgboost_booster_model_and_signature'), marks=pytest.mark.slow),
        pytest.param(lf('xgboost_xgb_model_and_signature'), marks=pytest.mark.slow),
        pytest.param(lf('sklearn_model_and_signature'), marks=pytest.mark.slow),
        # pytest.param(lf('torch_model_and_signature'), marks=pytest.mark.slow),
        # pytest.param(lf('tensorflow_model_and_signature'), marks=pytest.mark.slow),
        # pytest.param(lf('statsmodels_model_and_signature'), marks=pytest.mark.slow),
        # pytest.param(lf('prophet_model_and_signature'), marks=pytest.mark.slow),
        # pytest.param(lf('paddlepaddle_model_and_signature'), marks=pytest.mark.slow),
        # pytest.param(lf('spacy_model_and_signature'), marks=pytest.mark.slow),
        #
        # TODO: Mlflow don't support fastai model load by default. Need to add flavor
        # before test
        # pytest.param(lf('fastai_model_and_signature'), marks=pytest.mark.slow),
        # TODO: Configure CI to avoid OSError
        # pytest.param(lf('mxnet_model_and_signature'), marks=pytest.mark.slow),
        # pytest.param(lf('h2o_model_and_signature'), marks=pytest.mark.slow),
    ],
)
def test_log(model_and_signature, mlflow_run):
    model, input_example, output_example = model_and_signature

    log_model(model, name='model', input_example=input_example)

    model = mlflow.pyfunc.load_model(f'runs:/{mlflow_run.info.run_id}/model')
    assert assert_valide_model_output(model.predict(input_example), output_example)
