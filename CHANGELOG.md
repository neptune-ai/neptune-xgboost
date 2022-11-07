## neptune-xgboost 0.10.1

### Changes
- Moved `neptune-xgboost` package to `src` directory ([#12](https://github.com/neptune-ai/neptune-xgboost/pull/12))
- Moved to Poetry with package building ([#17](https://github.com/neptune-ai/neptune-xgboost/pull/17))

### Fixes
- Fixed import issue for callback - now it is possible to import as `from neptune_xgboost import NeptuneCallback`
  ([#14](https://github.com/neptune-ai/neptune-xgboost/pull/14))

## neptune-xgboost 0.10.0

### Changes
- Changed integrations utils to be imported from non-internal package ([#10](https://github.com/neptune-ai/neptune-xgboost/pull/10))

## neptune-xgboost 0.9.13

### Fixes
- Support `learning reate` for  `updater` types ([#8](https://github.com/neptune-ai/neptune-xgboost/pull/8))

## neptune-xgboost 0.9.12

### Fixes
- Do not log `learning rate` if value is unavailable ([#6](https://github.com/neptune-ai/neptune-xgboost/pull/6))

## neptune-xgboost 0.9.11

### Features
- Mechanism to prevent using legacy Experiments in new-API integrations ([#5](https://github.com/neptune-ai/neptune-xgboost/pull/5))
