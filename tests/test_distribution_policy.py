from scripts.verify_distribution import validate_member_names


def test_distribution_policy_allows_only_portable_package_members() -> None:
    members = (
        "signlab/__init__.py",
        "signlab/commands/data.py",
        "signlab/data/__init__.py",
        "signlab/models/tcn.py",
        "signlab-0.1.0/src/signlab/artifacts/metadata.py",
        "signlab-0.1.0.dist-info/METADATA",
    )

    assert validate_member_names(members) == ()


def test_distribution_policy_rejects_traversal_and_private_artifacts() -> None:
    assert validate_member_names(("../secret.txt",)) == (
        "archive contains a non-portable member path",
    )
    assert validate_member_names(("signlab/data/champion.onnx",)) == (
        "archive contains a private or generated artifact",
    )
    assert validate_member_names(("signlab-0.1.0/data/raw/participant.json",)) == (
        "archive contains a private or generated artifact",
    )
    assert validate_member_names(("models/champion.json",)) == (
        "archive contains a private or generated artifact",
    )
