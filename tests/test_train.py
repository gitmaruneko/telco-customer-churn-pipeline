import pandas as pd

from telco_churn.models import build_decision_tree_model, build_logistic_regression_model
from telco_churn.train import split_train_test


def test_train_test_split_is_reproducible_and_stratified() -> None:
    features = pd.DataFrame({"value": range(20)})
    target = pd.Series([0] * 10 + [1] * 10)

    first_split = split_train_test(features, target)
    second_split = split_train_test(features, target)

    for first, second in zip(first_split, second_split, strict=True):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)

    _, X_test, _, y_test = first_split
    assert len(X_test) == 4
    assert y_test.value_counts().to_dict() == {0: 2, 1: 2}


def test_both_model_builders_fit_and_predict() -> None:
    features = pd.DataFrame(
        {
            "service": ["A", "B"] * 6,
            "tenure": [1, 12, 3, 24, 5, 36, 7, 48, 9, 60, 11, 72],
        }
    )
    target = pd.Series([0, 1] * 6)

    for build_model in (build_logistic_regression_model, build_decision_tree_model):
        model = build_model(features)
        model.fit(features, target)

        assert model.predict(features).shape == target.shape
        assert model.predict_proba(features).shape == (len(features), 2)
