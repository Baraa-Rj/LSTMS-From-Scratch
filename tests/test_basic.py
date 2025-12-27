from lstms import __version__


def test_version_is_defined():
    assert isinstance(__version__, str)
    assert len(__version__) > 0
