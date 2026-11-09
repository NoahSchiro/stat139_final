from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.clean import clean
from src.eda import eda
from src.model import model


def parse_args() -> Namespace:
    """Parse the eda and model argument groups."""
    parser = ArgumentParser()
    parser.add_argument(
        "--clean-path",
        type=str,
        default="data/clean/session_tracks.csv",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    eda_parser = subparsers.add_parser("eda", help="Do EDA")
    eda_parser.set_defaults(func=eda)

    model_parser = subparsers.add_parser("model", help="Build a model")
    model_parser.set_defaults(func=model)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean(Path(args.clean_path))
    args.func(args)


if __name__ == "__main__":
    main()
