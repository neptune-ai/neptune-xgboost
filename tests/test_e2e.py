import pytest
import xgboost as xgb
from neptune.new.integrations.xgboost import NeptuneCallback

try:
    # neptune-client=0.9.0+ package structure
    from neptune.new import init_run
except ImportError:
    # neptune-client>=1.0.0 package structure
    from neptune import init_run


@pytest.mark.parametrize("log_tree", [None, [0, 1, 2, 3]])
def test_e2e(dataset, log_tree):
    # Start a run
    run = init_run()

    # Create a NeptuneCallback instance
    neptune_callback = NeptuneCallback(run=run, log_tree=log_tree)

    X_train, X_test, y_train, y_test = dataset

    data_train = xgb.DMatrix(X_train, label=y_train)

    # Define model parameters
    model_params = {
        "eta": 0.7,
        "gamma": 0.001,
        "max_depth": 9,
    }

    # Train the model and log metadata to the run in Neptune
    xgb.train(
        params=model_params,
        dtrain=data_train,
        callbacks=[neptune_callback],
    )
    run.wait()
    validate_results(run, log_tree, base_namespace="training")


@pytest.mark.parametrize("log_tree", [None, [0, 1, 2, 3]])
def test_e2e_using_namespace(dataset, log_tree):
    # Start a run
    run = init_run()

    # Create a NeptuneCallback instance
    neptune_callback = NeptuneCallback(run=run, base_namespace="my_namespace", log_tree=log_tree)

    X_train, X_test, y_train, y_test = dataset

    data_train = xgb.DMatrix(X_train, label=y_train)

    # Define model parameters
    model_params = {
        "eta": 0.7,
        "gamma": 0.001,
        "max_depth": 9,
    }

    # Train the model and log metadata to the run in Neptune
    xgb.train(
        params=model_params,
        dtrain=data_train,
        callbacks=[neptune_callback],
    )
    run.wait()
    validate_results(run, log_tree, base_namespace="my_namespace")


def validate_results(run, log_tree, base_namespace):
    assert run.exists(f"{base_namespace}/booster_config")
    assert run.exists(f"{base_namespace}/booster_config/learner")
    assert run.exists(f"{base_namespace}/booster_config/version")

    assert run.exists(f"{base_namespace}/epoch")
    assert run.exists(f"{base_namespace}/learning_rate")
    assert run.exists(f"{base_namespace}/plots")
    assert run[f"{base_namespace}/plots/importance"].fetch_extension() == "png"

    if log_tree:
        assert run.exists(f"{base_namespace}/plots/trees")
    else:
        assert not run.exists(f"{base_namespace}/plots/trees")
