from importlib.metadata import metadata, version

from signlab import __version__


def test_package_version_has_one_runtime_source() -> None:
    assert __version__ == "0.1.0"
    assert version("signlab") == __version__


def test_distribution_metadata_is_complete() -> None:
    package_metadata = metadata("signlab")

    assert package_metadata["Name"] == "signlab"
    assert package_metadata["License-Expression"] == "MIT"
    assert package_metadata["Requires-Python"] == "<3.13,>=3.12"
    requirements = tuple(package_metadata.get_all("Requires-Dist") or ())
    assert not any(
        requirement.casefold().startswith(
            (
                "alembic",
                "dvc",
                "dvc-s3",
                "keras",
                "mlflow-skinny",
                "onnx",
                "onnxruntime",
                "scikit-learn",
                "sqlalchemy",
                "tensorflow",
                "tf2onnx",
            )
        )
        and "extra ==" not in requirement.casefold()
        for requirement in requirements
    )
    assert set(package_metadata.get_all("Provides-Extra") or ()) == {
        "experiments",
        "extraction",
        "legacy-compatibility",
    }
