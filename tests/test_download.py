from msfiddle.download import checkpoint_release_for_package_version


def test_checkpoint_release_for_2_0_1_reuses_2_0_0_assets():
    assert checkpoint_release_for_package_version("2.0.1") == "v2.0.0"


def test_checkpoint_release_defaults_to_matching_package_version():
    assert checkpoint_release_for_package_version("2.0.0") == "v2.0.0"
    assert checkpoint_release_for_package_version("2.1.0") == "v2.1.0"
