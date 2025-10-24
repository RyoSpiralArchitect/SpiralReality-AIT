import importlib
import os
import sys

import numpy as real_numpy
import pytest

from spiralreality_AIT_onepass_aifcore_integrated.integrated import np_stub


@pytest.fixture
def restore_np_stub(monkeypatch):
    original_env = os.environ.get("SPIRAL_NUMERIC_FORCE_STUB")

    def _reload(force: bool | None = None):
        if force is None:
            if original_env is None:
                monkeypatch.delenv("SPIRAL_NUMERIC_FORCE_STUB", raising=False)
            else:
                monkeypatch.setenv("SPIRAL_NUMERIC_FORCE_STUB", original_env)
        elif force:
            monkeypatch.setenv("SPIRAL_NUMERIC_FORCE_STUB", "1")
        else:
            monkeypatch.delenv("SPIRAL_NUMERIC_FORCE_STUB", raising=False)
        for mod in [
            "spiralreality_AIT_onepass_aifcore_integrated.integrated.np_stub",
            "spiralreality_AIT_onepass_aifcore_integrated.integrated._np_stub_numpy",
            "spiralreality_AIT_onepass_aifcore_integrated.integrated._np_stub_purepy",
        ]:
            sys.modules.pop(mod, None)
        return importlib.import_module(
            "spiralreality_AIT_onepass_aifcore_integrated.integrated.np_stub"
        )

    try:
        yield _reload
    finally:
        if original_env is None:
            os.environ.pop("SPIRAL_NUMERIC_FORCE_STUB", None)
        else:
            os.environ["SPIRAL_NUMERIC_FORCE_STUB"] = original_env
        for mod in [
            "spiralreality_AIT_onepass_aifcore_integrated.integrated.np_stub",
            "spiralreality_AIT_onepass_aifcore_integrated.integrated._np_stub_numpy",
            "spiralreality_AIT_onepass_aifcore_integrated.integrated._np_stub_purepy",
        ]:
            sys.modules.pop(mod, None)
        importlib.import_module(
            "spiralreality_AIT_onepass_aifcore_integrated.integrated.np_stub"
        )


def test_linalg_inv_matches_numpy():
    mat = np_stub.array([[4.0, 2.0, 0.0], [2.0, 4.0, 2.0], [0.0, 2.0, 4.0]])
    stub_result = np_stub.linalg.inv(mat)
    real_result = real_numpy.linalg.inv(real_numpy.array(mat.to_list(), dtype=float))
    assert stub_result.to_list() == pytest.approx(real_result.tolist(), abs=1e-10)


def test_linalg_inv_partial_pivot_matches_numpy_pure_stub(restore_np_stub):
    stub_module = restore_np_stub(force=True)
    try:
        mat = stub_module.array([[0.0, 1.0], [1.0, 1.0]])
        stub_result = stub_module.linalg.inv(mat)
        real_result = real_numpy.linalg.inv(real_numpy.array([[0.0, 1.0], [1.0, 1.0]], dtype=float))
        assert stub_result.to_list() == pytest.approx(real_result.tolist(), abs=1e-9)
    finally:
        restore_np_stub(force=False)


def test_dot_and_matmul_shape_handling():
    a_vec = np_stub.array([1.0, 2.0, 3.0])
    b_vec = np_stub.array([4.0, 5.0, 6.0])
    assert np_stub.dot(a_vec, b_vec) == pytest.approx(32.0)

    mat = np_stub.array([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    prod = np_stub.matmul(mat, b_vec)
    assert prod.to_list() == pytest.approx([10.0, 13.0])

    with pytest.raises(ValueError) as excinfo:
        np_stub.dot(np_stub.array([[1.0]]), np_stub.array([[[1.0]]]))
    assert "supports up to 2D" in str(excinfo.value)


def test_safe_exp_clamps_large_values():
    assert np_stub.safe_exp(1_000.0) == pytest.approx(real_numpy.exp(np_stub.SAFE_EXP_CLIP))
    assert np_stub.safe_exp(-1_000.0) == pytest.approx(real_numpy.exp(-np_stub.SAFE_EXP_CLIP))


def test_random_normal_matches_numpy_rng():
    stub_rng = np_stub.random.default_rng(123)
    stub_vals = stub_rng.normal(size=(4, 3)).to_list()
    real_rng = real_numpy.random.default_rng(123)
    real_vals = real_rng.normal(size=(4, 3))
    assert stub_vals == pytest.approx(real_vals.tolist())


def test_slogdet_matches_numpy():
    mat = np_stub.array([[2.0, 1.0], [1.0, 2.0]])
    stub_sign, stub_logdet = np_stub.linalg.slogdet(mat)
    real_sign, real_logdet = real_numpy.linalg.slogdet(real_numpy.array(mat.to_list(), dtype=float))
    assert stub_sign == pytest.approx(real_sign)
    assert stub_logdet == pytest.approx(real_logdet)
