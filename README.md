# Neptune + XGBoost integration

Experiment tracking, model registry, data versioning, and live model monitoring for XGBoost trained models.

## What will you get with this integration?

* Log, display, organize, and compare ML experiments in a single place
* Version, store, manage, and query trained models, and model building metadata
* Record and monitor model training, evaluation, or production runs live

## What will be logged to Neptune?

* metrics,
* parameters,
* learning rate,
* pickled model,
* visualizations (feature importance chart and tree visualizations),
* hardware consumption (CPU, GPU, Memory),
* stdout and stderr logs,
* training code and Git commit information,
* [other metadata](https://docs.neptune.ai/logging/what_you_can_log)

![image](https://user-images.githubusercontent.com/97611089/160614588-5d839a11-e2f9-4eed-a3d1-39314ebdb1ea.png)
*Example dashboard with train-valid metrics and selected parameters*


## Resources

* [Documentation](https://docs.neptune.ai/integrations/xgboost)
* [Code example on GitHub](https://github.com/neptune-ai/examples/blob/main/integrations-and-supported-tools/xgboost/scripts/Neptune_XGBoost_train.py)
* [Example of a run logged in the Neptune app](https://app.neptune.ai/o/common/org/xgboost-integration/e/XGBOOST-84/dashboard/train-e395296a-4f3d-4a58-ab88-6ef06bbac657)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/xgboost/notebooks/Neptune_XGBoost.ipynb)

## Example

On the command line:

```
pip install xgboost>=1.3.0 neptune-xgboost
```

In Python:

```python
import neptune
import xgboost as xgb
from neptune.integrations.xgboost import NeptuneCallback

# Start a run
run = neptune.init_run(
    project="common/xgboost-integration",
    api_token=neptune.ANONYMOUS_API_TOKEN,
)

# Create a NeptuneCallback instance
neptune_callback = NeptuneCallback(run=run, log_tree=[0, 1, 2, 3])

# Prepare datasets
...
data_train = xgb.DMatrix(X_train, label=y_train)

# Define model parameters
model_params = {
    "eta": 0.7,
    "gamma": 0.001,
    "max_depth": 9,
    ...
}

# Train the model and log metadata to the run in Neptune
xgb.train(
    params=model_params,
    dtrain=data_train,
    callbacks=[neptune_callback],
)
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting_help)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
