from signlab import __version__
from signlab.cli import build_parser


def test_package_version_is_exposed() -> None:
    assert __version__ == "0.1.0"


def test_cli_parser_has_expected_program_name() -> None:
    assert build_parser().prog == "signlab"
