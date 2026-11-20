import pickle

import numpy as np
import pytest

from qrc.reset import Reset

U = np.random.default_rng(0).uniform(-1, 1, 400_000)
SHAPES = [("linear", {}), ("magnitude", {}), ("quadratic", {}), ("sigmoid", {"k": 10.0, "u0": 0.5}),
          ("legendre", {"coefs": [0.3, -1.0, 0.5]})]


@pytest.mark.parametrize("kind,shape", SHAPES)
def test_shape_zero_mean_unit_variance(kind, shape):
    g = Reset(kind, 0.5, **shape).g(U)
    assert abs(g.mean()) < 0.01 and abs(g.std() - 1) < 0.01


@pytest.mark.parametrize("kind,shape", SHAPES)
def test_mean_reset_is_p0_and_a_is_std(kind, shape):
    f = Reset.from_fraction(kind, 0.4, 0.8, **shape)
    p = f(U)
    assert abs(p.mean() - 0.4) < 0.01 and abs(p.std() - f.a) < 0.01


@pytest.mark.parametrize("kind,shape", SHAPES)
@pytest.mark.parametrize("frac", [-1.0, 1.0])
def test_full_fraction_touches_boundary(kind, shape, frac):
    lo, hi = Reset.from_fraction(kind, 0.3, frac, **shape).p_range()
    assert lo >= 0 and hi <= 1
    assert min(lo, 1 - hi) < 1e-6                     # the full allowed range reaches 0 or 1


def test_known_shapes():
    assert np.allclose(Reset("linear", 0.5, 0.1).g(np.array([-1, 0, 1])), np.sqrt(3) * np.array([-1, 0, 1]))
    assert np.allclose(Reset("magnitude", 0.5, 0.1).g(np.array([0.5])), 0.0)
    assert np.isclose(Reset("linear", 0.5).a_max(+1), 0.5 / np.sqrt(3))


def test_out_of_range_rejected():
    with pytest.raises(ValueError, match="leaves"):
        Reset("linear", 0.9, 0.2)
    with pytest.raises(ValueError, match="needs parameters"):
        Reset("sigmoid", 0.5, 0.1, k=5.0)
    with pytest.raises(ValueError, match="unknown"):
        Reset("cubic", 0.5)


def test_constant_and_picklable():
    f = Reset("constant", 0.7)
    assert np.all(f(U[:10]) == 0.7)
    g = pickle.loads(pickle.dumps(Reset("sigmoid", 0.5, 0.1, k=8.0, u0=0.4)))
    assert np.allclose(g(U[:10]), Reset("sigmoid", 0.5, 0.1, k=8.0, u0=0.4)(U[:10]))
