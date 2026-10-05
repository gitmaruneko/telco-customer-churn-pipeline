import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "model.toml"


@dataclass(frozen=True)
class ModelSettings:
    test_size: float
    split_random_state: int
    logistic_regression_max_iter: int
    decision_tree_random_state: int
    dataset_path: Path
    output_directory: Path
    predictions_filename: str
    comparison_filename: str


def load_model_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> ModelSettings:
    """Load model and output settings from a TOML file."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    split_settings = config["split"]
    model_settings = config["models"]
    input_settings = config["input"]
    output_settings = config["output"]
    test_size = float(split_settings["test_size"])
    logistic_regression_max_iter = int(model_settings["logistic_regression"]["max_iter"])

    if not 0 < test_size < 1:
        raise ValueError("split.test_size must be between 0 and 1")
    if logistic_regression_max_iter < 1:
        raise ValueError("models.logistic_regression.max_iter must be positive")

    output_directory = Path(output_settings["directory"])
    if not output_directory.is_absolute():
        output_directory = PROJECT_ROOT / output_directory

    dataset_path = Path(input_settings["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    return ModelSettings(
        test_size=test_size,
        split_random_state=int(split_settings["random_state"]),
        logistic_regression_max_iter=logistic_regression_max_iter,
        decision_tree_random_state=int(model_settings["decision_tree"]["random_state"]),
        dataset_path=dataset_path,
        output_directory=output_directory,
        predictions_filename=str(output_settings["predictions_filename"]),
        comparison_filename=str(output_settings["comparison_filename"]),
    )
