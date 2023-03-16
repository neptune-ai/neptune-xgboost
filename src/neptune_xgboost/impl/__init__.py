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
    "__version__",
]

import json
import subprocess
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib import image

try:
    # neptune-client>=1.0.0 package structure
    import neptune
    from neptune import Run
    from neptune.handler import Handler
    from neptune.integrations.utils import (
        expect_not_an_experiment,
        verify_type,
    )
    from neptune.utils import stringify_unsupported

except ImportError:
    import neptune.new as neptune
    from neptune.new.metadata_containers import Run
    from neptune.new.handler import Handler
    from neptune.new.integrations.utils import (
        expect_not_an_experiment,
        verify_type,
    )
    from neptune.new.utils import stringify_unsupported

from neptune_xgboost.impl.version import __version__

INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-xgboost"


class NeptuneCallback(xgb.callback.TrainingCallback):
    """Neptune callback for logging metadata during XGBoost model training.

    This callback logs metrics, all parameters, learning rate, pickled model, and visualizations.
    If early stopping is activated, "best_score" and "best_iteration" are also logged.

    Metrics are logged for every dataset in the `evals` list and for every metric specified.

    The callback works with the `xgboost.train()` and `xgboost.cv()` functions,
    and with `model.fit()` from the the scikit-learn API.

    Note: This callback requires `xgboost>=1.3.0`.

    Args:
        run: Neptune run object. You can also pass a namespace handler object;
            for example, run["test"], in which case all metadata is logged under
            the "test" namespace inside the run.
        base_namespace: Root namespace where all metadata logged by the callback is stored.
        log_model: Whether to log model as pickled file at the end of training.
        log_importance: Whether to log feature importance charts at the end of training.
        max_num_features: Max number of top features on the importance charts.
            Works only if log_importance is set to True. If None, all features are displayed.
        log_tree: Indexes of the target trees to log as charts.
            Requires graphviz to be installed.
        tree_figsize: Size of the visualized tree image.
            Increase this in case you work with large trees. Works only if log_tree is not None.

    Example:
        import neptune
        from neptune.integrations.xgboost import NeptuneCallback

        run = neptune.init_run()
        neptune_callback = NeptuneCallback(run=run)
        xgb.train( ..., callbacks=[neptune_callback])

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/xgboost/
        API reference: https://docs.neptune.ai/api/integrations/xgboost/
    """

    def __init__(
        self,
        run,
        base_namespace="training",
        log_model=True,
        log_importance=True,
        max_num_features=None,
        log_tree=None,
        tree_figsize=30,
    ):

        expect_not_an_experiment(run)
        verify_type("run", run, (Run, Handler))
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
                message = (
                    "Graphviz executables not found, so trees will not be logged. "
                    "Make sure the Graphviz executables are on your systems' PATH"
                )
                warnings.warn(message)

        root_obj = self.run
        if isinstance(self.run, Handler):
            root_obj = self.run.get_root_object()

        root_obj[INTEGRATION_VERSION_KEY] = __version__

    def before_training(self, model):
        if hasattr(model, "cvfolds"):
            self.cv = True
        return model

    def after_training(self, model):
        # model structure is different for "cv" and "train" functions that you use to train xgb model
        if self.cv:
            for i, fold in enumerate(model.cvfolds):
                self.run[f"fold_{i}/booster_config"] = stringify_unsupported(json.loads(fold.bst.save_config()))
        else:
            self.run["booster_config"] = stringify_unsupported(json.loads(model.save_config()))
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
        self.run["epoch"].append(epoch)
        self._log_metrics(evals_log)
        self._log_learning_rate(model)
        return False

    def _log_metrics(self, evals_log):
        for stage, metrics_dict in evals_log.items():
            for metric_name, metric_values in evals_log[stage].items():
                if self.cv:
                    mean, std = metric_values[-1]
                    self.run[stage][metric_name]["mean"].append(mean)
                    self.run[stage][metric_name]["std"].append(std)
                else:
                    self.run[stage][metric_name].append(metric_values[-1])

    def _log_learning_rate(self, model):
        if self.cv:
            config = json.loads(model.cvfolds[0].bst.save_config())
        else:
            config = json.loads(model.save_config())

        lr = None
        updater = config["learner"]["gradient_booster"]["updater"]
        updater_types = [
            "grow_colmaker",
            "grow_histmaker",
            "grow_local_histmaker",
            "grow_quantile_histmaker",
            "grow_gpu_hist",
            "sync",
            "refresh",
            "prune",
        ]
        for updater_type in updater_types:
            if updater_type in updater:
                lr = updater[updater_type]["train_param"]["learning_rate"]
                break

        if lr is not None:
            self.run["learning_rate"].append(float(lr))
