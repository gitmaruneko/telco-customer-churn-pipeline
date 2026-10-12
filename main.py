from collections.abc import Sequence

from telco_churn.train import main as run_cli


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Telco customer churn command-line application."""
    run_cli(argv)


if __name__ == "__main__":
    main()
