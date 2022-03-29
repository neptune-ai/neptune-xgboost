# XGBoost + Neptune Integration

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
* stdout and stderr logs, and
* training code and git commit information

![image](https://user-images.githubusercontent.com/97611089/160614588-5d839a11-e2f9-4eed-a3d1-39314ebdb1ea.png)
*Example dashboard with train-valid metrics and selected parameters*


## Resources

* [Documentation](https://docs.neptune.ai/integrations-and-supported-tools/model-training/xgboost)
* [Code example on GitHub](https://github.com/neptune-ai/examples/blob/main/integrations-and-supported-tools/xgboost/scripts/Neptune_XGBoost_train.py)
* [Example of a run logged in the Neptune app](https://app.neptune.ai/o/common/org/xgboost-integration/e/XGBOOST-84/dashboard/train-e395296a-4f3d-4a58-ab88-6ef06bbac657)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/xgboost/notebooks/Neptune_XGBoost.ipynb)
