from scripts.verify_distribution import validate_member_names


def test_distribution_policy_allows_only_portable_package_members() -> None:
    members = (
        "signlab/__init__.py",
        "signlab/commands/data.py",
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
