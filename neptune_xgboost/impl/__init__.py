#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

__all__ = [
    "NeptuneCallback",
]

import json
import subprocess
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib import image

from neptune_xgboost import __version__

try:
    # neptune-client=0.9.0 package structure
    import neptune.new as neptune
    from neptune.new.internal.utils import verify_type
except ImportError:
    # neptune-client=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type

INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-xgboost"


class NeptuneCallback(xgb.callback.TrainingCallback):
    """Neptune callback for logging metadata during XGBoost model training.

    See guide with examples in the `Neptune-XGBoost docs`_.

    This callback logs metrics, all parameters, learning rate, pickled model, visualizations.
    If early stopping is activated "best_score" and "best_iteration" is also logged.

    All metadata are collected under the common namespace that you can specify.
    See: ``base_namespace`` argument (defaults to "training").

    Metrics are logged for every dataset in the ``evals`` list and for every metric specified.
    For example with ``evals = [(dtrain, "train"), (dval, "valid")]`` and ``"eval_metric": ["mae", "rmse"]``,
    4 metrics are created::

        "train/mae"
        "train/rmse"
        "valid/mae"
        "valid/rmse"

    Visualizations are feature importances and trees.

    Callback works with ``xgboost.train()`` and ``xgboost.cv()`` functions, and with the sklearn API ``model.fit()``.

    Note:
        This callback works with ``xgboost>=1.3.0``. This release introduced new style Python callback API.

    Note:
        You can use public ``api_token="ANONYMOUS"`` and set ``project="common/xgboost-integration"``
        for testing without registration.

    Args:
        run (:obj:`neptune.new.run.Run`): Neptune run object.
            A run in Neptune is a representation of all metadata that you log to Neptune.
            Learn more in `run docs`_.
        base_namespace(:obj:`str`, optional): Defaults to "training".
            Root namespace. All metadata will be logged inside.
        log_model (bool): Defaults to True. Log model as pickled file at the end of training.
        log_importance (bool): Defaults to True. Log feature importance charts at the end of training.
        max_num_features (int): Defaults to None. Max number of top features on the importance charts.
            Works only if ``log_importance`` is set to ``True``. If None, all features will be displayed.
            See `xgboost.plot_importance`_ for details.
        log_tree (list): Defaults to None. Indices of the target trees to log as charts.
            This requires graphviz to work. Learn about setup in the `Neptune-XGBoost installation`_ docs.
            See `xgboost.to_graphviz`_ for details.
        tree_figsize (int): Defaults to 30, Control size of the visualized tree image.
            Increase this in case you work with large trees. Works only if ``log_tree`` is list.

    Examples:
        For more examples visit `example scripts`_.
        Full script that does model training and logging of the metadata::

            import neptune.new as neptune
            import xgboost as xgb
            from neptune.new.integrations.xgboost import NeptuneCallback
            from sklearn.datasets import load_boston
            from sklearn.model_selection import train_test_split

            # Create run
            run = neptune.init(
                project="common/xgboost-integration",
                api_token="ANONYMOUS",
                name="xgb-train",
                tags=["xgb-integration", "train"]
            )

            # Create neptune callback
            neptune_callback = NeptuneCallback(run=run, log_tree=[0, 1, 2, 3])

            # Prepare data
            X, y = load_boston(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_test, label=y_test)

            # Define parameters
            model_params = {
                "eta": 0.7,
                "gamma": 0.001,
                "max_depth": 9,
                "objective": "reg:squarederror",
                "eval_metric": ["mae", "rmse"]
            }
            evals = [(dtrain, "train"), (dval, "valid")]
            num_round = 57

            # Train the model and log metadata to the run in Neptune
            xgb.train(
                params=model_params,
                dtrain=dtrain,
                num_boost_round=num_round,
                evals=evals,
                callbacks=[
                    neptune_callback,
                    xgb.callback.LearningRateScheduler(lambda epoch: 0.99**epoch),
                    xgb.callback.EarlyStopping(rounds=30)
                ],
            )

    .. _Neptune-XGBoost docs:
        https://docs.neptune.ai/integrations-and-supported-tools/model-training/xgboost
       _Neptune-XGBoost installation:
        https://docs.neptune.ai/integrations-and-supported-tools/model-training/xgboost#install-requirements
       _run docs:
        https://docs.neptune.ai/api-reference/run
       _xgboost.plot_importance:
        https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.plot_importance
       _xgboost.to_graphviz:
        https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.to_graphviz
       _example scripts:
        https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/xgboost/scripts
    """

    def __init__(self,
                 run,
                 base_namespace="training",
                 log_model=True,
                 log_importance=True,
                 max_num_features=None,
                 log_tree=None,
                 tree_figsize=30):

        verify_type("run", run, neptune.Run)
        verify_type("base_namespace", base_namespace, str)
        log_model is not None and verify_type("log_model", log_model, bool)
        log_importance is not None and verify_type("log_importance", log_importance, bool)
        max_num_features is not None and verify_type("max_num_features", max_num_features, int)
        log_tree is not None and verify_type("log_tree", log_tree, list)
        verify_type("tree_figsize", tree_figsize, int)

        self.run = run[base_namespace]
        self.log_model = log_model
        self.log_importance = log_importance
        self.max_num_features = max_num_features
        self.log_tree = log_tree
        self.cv = False
        self.tree_figsize = tree_figsize

        if self.log_tree:
            try:
                subprocess.call(["dot", "-V"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                self.log_tree = None
                message = "Graphviz executables not found, so trees will not be logged. " \
                          "Make sure the Graphviz executables are on your systems' PATH"
                warnings.warn(message)

        run[INTEGRATION_VERSION_KEY] = __version__

    def before_training(self, model):
        if hasattr(model, "cvfolds"):
            self.cv = True
        return model

    def after_training(self, model):
        # model structure is different for "cv" and "train" functions that you use to train xgb model
        if self.cv:
            for i, fold in enumerate(model.cvfolds):
                self.run[f"fold_{i}/booster_config"] = json.loads(fold.bst.save_config())
        else:
            self.run["booster_config"] = json.loads(model.save_config())
            if "best_score" in model.attributes().keys():
                self.run["early_stopping/best_score"] = model.attributes()["best_score"]
            if "best_iteration" in model.attributes().keys():
                self.run["early_stopping/best_iteration"] = model.attributes()["best_iteration"]

        self._log_importance(model)
        self._log_trees(model)
        self._log_model(model)
        return model

    def _log_importance(self, model):
        if self.log_importance:
            # for "cv" log importance chart per fold
            if self.cv:
                for i, fold in enumerate(model.cvfolds):
                    importance = xgb.plot_importance(fold.bst, max_num_features=self.max_num_features)
                    self.run[f"fold_{i}/plots/importance"].upload(neptune.types.File.as_image(importance.figure))
                plt.close("all")
            else:
                importance = xgb.plot_importance(model, max_num_features=self.max_num_features)
                self.run["plots/importance"].upload(neptune.types.File.as_image(importance.figure))
                plt.close("all")

    def _log_trees(self, model):
        if self.log_tree is not None:
            # for "cv" log trees for each cv fold (different model is trained on each fold)
            if self.cv:
                for i, fold in enumerate(model.cvfolds):
                    trees = []
                    for j in self.log_tree:
                        tree = xgb.to_graphviz(fold.bst, num_trees=j)
                        _, ax = plt.subplots(1, 1, figsize=(self.tree_figsize, self.tree_figsize))
                        s = BytesIO()
                        s.write(tree.pipe(format="png"))
                        s.seek(0)
                        ax.imshow(image.imread(s))
                        ax.axis("off")
                        trees.append(neptune.types.File.as_image(ax.figure))
                    self.run[f"fold_{i}/plots/trees"] = neptune.types.FileSeries(trees)
                    plt.close("all")
            else:
                trees = []
                for j in self.log_tree:
                    tree = xgb.to_graphviz(model, num_trees=j)
                    _, ax = plt.subplots(1, 1, figsize=(self.tree_figsize, self.tree_figsize))
                    s = BytesIO()
                    s.write(tree.pipe(format="png"))
                    s.seek(0)
                    ax.imshow(image.imread(s))
                    ax.axis("off")
                    trees.append(neptune.types.File.as_image(ax.figure))
                self.run["plots/trees"] = neptune.types.FileSeries(trees)
                plt.close("all")

    def _log_model(self, model):
        if self.log_model:
            # for "cv" log model per fold
            if self.cv:
                for i, fold in enumerate(model.cvfolds):
                    self.run[f"fold_{i}/pickled_model"].upload(neptune.types.File.as_pickle(fold.bst))
            else:
                self.run["pickled_model"].upload(neptune.types.File.as_pickle(model))

    def before_iteration(self, model, epoch: int, evals_log) -> bool:
        # False to indicate training should not stop.
        return False

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        self.run["epoch"].log(epoch)
        self._log_metrics(evals_log)
        self._log_learning_rate(model)
        return False

    def _log_metrics(self, evals_log):
        for stage, metrics_dict in evals_log.items():
            for metric_name, metric_values in evals_log[stage].items():
                if self.cv:
                    mean, std = metric_values[-1]
                    self.run[stage][metric_name]["mean"].log(mean)
                    self.run[stage][metric_name]["std"].log(std)
                else:
                    self.run[stage][metric_name].log(metric_values[-1])

    def _log_learning_rate(self, model):
        if self.cv:
            config = json.loads(model.cvfolds[0].bst.save_config())
        else:
            config = json.loads(model.save_config())
        lr = config["learner"]["gradient_booster"]["updater"]["grow_colmaker"]["train_param"]["learning_rate"]
        self.run["learning_rate"].log(float(lr))
