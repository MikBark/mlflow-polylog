import json
from pathlib import Path

import yaml


def check_path_exists_and_is_dir(path: Path):
    assert path.exists() and path.is_dir(), (
        f'Path {path} does not exist or is not a directory'
    )


def check_required_files(path: Path, required_files):
    for file in required_files:
        assert (path / file).exists(), f'Required file {file} is missing in {path}'


def check_mlmodel_file_format(mlmodel_path: Path):
    try:
        with open(mlmodel_path) as f:
            mlmodel_content = yaml.safe_load(f)
    except Exception as e:
        assert False, f'MLmodel file is not a valid YAML: {e}'

    assert isinstance(mlmodel_content, dict), 'MLmodel content is not a dictionary'
    assert 'flavors' in mlmodel_content, "MLmodel file does not contain 'flavors' key"


def check_optional_json_files(path: Path, optional_files):
    for opt_file in optional_files[:2]:
        opt_path = path / opt_file
        if opt_path.exists():
            try:
                with open(opt_path) as f:
                    json.load(f)
            except Exception as e:
                assert False, f'Optional file {opt_file} is not a valid JSON: {e}'


def check_environment_variables_file(path: Path, env_file: str):
    env_vars_path = path / env_file
    if env_vars_path.exists():
        assert env_vars_path.is_file(), f'{env_file} should be a file'


def assert_mlflow_model_logged(path: Path):
    required_files = [
        'MLmodel',
        'model.pkl',
        'conda.yaml',
        'python_env.yaml',
        'requirements.txt',
    ]
    optional_files = [
        'input_example.json',
        'serving_input_example.json',
        'environment_variables.txt',
    ]

    check_path_exists_and_is_dir(path)
    check_required_files(path, required_files)
    check_mlmodel_file_format(path / 'MLmodel')
    check_optional_json_files(path, optional_files)
    check_environment_variables_file(path, optional_files[2])

    return True
