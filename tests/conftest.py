import pytest


@pytest.fixture(autouse=True)
def _private_voices(tmp_path_factory, monkeypatch):
    """Never read or write the user's saved voices."""
    monkeypatch.setenv("TUISCRIBE_VOICES_DIR", str(tmp_path_factory.mktemp("voices")))
