from __future__ import annotations

import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import np_stub


def test_parse_bool_env_understands_synonyms():
    assert np_stub._parse_bool_env("1") is True
    assert np_stub._parse_bool_env("TRUE") is True
    assert np_stub._parse_bool_env("0") is False
    assert np_stub._parse_bool_env("off") is False
    assert np_stub._parse_bool_env("auto") is None


def test_backend_call_propagates_in_strict_mode(monkeypatch: pytest.MonkeyPatch):
    class FailingBackend:
        def mean(self, *_args, **_kwargs):
            raise RuntimeError("backend failure")

    monkeypatch.setattr(np_stub, "_ACCEL_BACKEND", FailingBackend())
    monkeypatch.setattr(np_stub, "_STRICT_BACKEND", True)

    with pytest.raises(RuntimeError):
        np_stub._backend_call("mean", [1.0, 2.0, 3.0], axis=None, keepdims=False)


def test_backend_call_falls_back_when_not_strict(monkeypatch: pytest.MonkeyPatch):
    class FailingBackend:
        def sum(self, *_args, **_kwargs):
            raise RuntimeError("backend failure")

    monkeypatch.setattr(np_stub, "_ACCEL_BACKEND", FailingBackend())
    monkeypatch.setattr(np_stub, "_STRICT_BACKEND", False)

    result = np_stub._backend_call("sum", [1.0, 2.0, 3.0], axis=None, keepdims=False)
    assert result is None

