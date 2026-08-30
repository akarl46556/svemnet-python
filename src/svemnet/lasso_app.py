"""Separate desktop/CLI entry point for single-model lasso and elastic net."""

from .screening_app import main as _main


def main(argv=None):
    return _main(argv, base_model=True)


if __name__ == "__main__":
    raise SystemExit(main())
