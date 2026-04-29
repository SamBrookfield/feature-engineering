
# FEATURES FROM A REGISTRY

This library provides a modular framework for feature engineering, including feature design, building, and selection. It enables consistent feature definitions across datasets, reproducible transformations, and structured feature selection workflows for machine learning pipelines.

See examples in 'examples_notebook.ipynb'



# PROJECT STRUCTURE
### `src/feature_design/`
Provide a framework for a registry of features across datasets that captures definition mathematically and understandability

### `src/feature_building/`
Provide a framework that builds a dataset of complete features from a dataset and a feature registry

### `src/feature_selection/`
Provide a framework and variety of different methods for tagging features in a feature registry for later identification and selecting final columns for modelling



# QUICKSTART
Establish registry and design features:

```python
from src.feature_design import RegistryConfig, FeatureRegistry, FeatureSpec, Direction

CREDIT_CONFIG = RegistryConfig(
    families = {
        "CREDIT_RISK", "REPAYMENT", "INCOME_BURDEN",
        "BEHAVIOURAL", "DEMOGRAPHIC",
    },
    sources = {
        "TEST_DATASET", "DERVIED"
    },
)
REGISTRY = FeatureRegistry(config=CREDIT_CONFIG)

REGISTRY.register_many([
    FeatureSpec('SPEND_INCOME_RATIO', "INCOME_BURDEN", "TEST_DATASET",
                'Monthly spend as a fraction of monthly income — proxy for affordability pressure',
                formula='avg_monthly_spend / (income / 12)',
                compute=lambda df: df['avg_monthly_spend'] / (df['income'] / 12).replace(0, np.nan),
                actionable=True, direction=Direction.POSITIVE,
                tags=['affordability', 'engineered']),
    ...
```

Call registry when building features: 
```
from src.feature_builders import sanitise_columns, build_features, downcast_dtypes

test_dataset_sanitised = sanitise_columns(test_dataset)
test_dataset_features = build_features(
    df = test_dataset_sanitised,
    registry = REGISTRY,
    source = "TEST_DATASET",
)
test_dataset_features = downcast_dtypes(test_dataset_features) 
```
See full examples in `examples_notebook.ipynb`.

