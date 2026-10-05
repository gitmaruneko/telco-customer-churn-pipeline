from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from telco_churn.preprocessing import build_preprocessor


def build_logistic_regression_model(features, *, max_iter: int = 1000) -> Pipeline:
    """Build a preprocessing and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", LogisticRegression(max_iter=max_iter)),
        ]
    )


def build_decision_tree_model(features, *, random_state: int = 42) -> Pipeline:
    """Build a preprocessing and Decision Tree pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", DecisionTreeClassifier(random_state=random_state)),
        ]
    )
