import numpy as np
from ga_trader.ta.core import rma, rsi

def test_rma_monotonic_seed():
    x = np.arange(1, 101).astype(float)
    y = rma(x, 14)
    assert np.isnan(y[:13]).all()
    assert np.isfinite(y[13:]).all()

def test_rsi_bounds():
    x = np.linspace(1, 100, 500)
    y = rsi(x, 14)
    m = y[np.isfinite(y)]
    assert (m >= 0).all() and (m <= 100).all()
