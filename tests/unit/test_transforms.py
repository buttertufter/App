"""Test statistical transformation functions."""

import numpy as np
import pytest

from equations.statistics.transforms import logistic_sigmoid, geo_mean, pct_clip


class TestLogisticSigmoid:
    """Tests for logistic sigmoid transformation."""
    
    def test_sigmoid_at_zero(self):
        """Sigmoid of 0 should be 0.5."""
        assert logistic_sigmoid(0.0) == 0.5
    
    def test_sigmoid_positive(self):
        """Sigmoid of positive values should be > 0.5."""
        assert logistic_sigmoid(1.0) > 0.5
        assert logistic_sigmoid(10.0) > 0.99
    
    def test_sigmoid_negative(self):
        """Sigmoid of negative values should be < 0.5."""
        assert logistic_sigmoid(-1.0) < 0.5
        assert logistic_sigmoid(-10.0) < 0.01
    
    def test_sigmoid_bounds(self):
        """Sigmoid should always be in [0, 1]."""
        for x in [-100, -10, -1, 0, 1, 10, 100]:
            result = logistic_sigmoid(x)
            assert 0 <= result <= 1
    
    def test_sigmoid_kappa_effect(self):
        """Higher kappa should create steeper curve."""
        result_low = logistic_sigmoid(0.5, kappa=1.0)
        result_high = logistic_sigmoid(0.5, kappa=10.0)
        assert result_high > result_low
    
    def test_sigmoid_array(self):
        """Sigmoid should work with arrays."""
        x = np.array([-1, 0, 1])
        result = logistic_sigmoid(x)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[1] == 0.5


class TestGeoMean:
    """Tests for geometric mean."""
    
    def test_geo_mean_simple(self):
        """Geometric mean of equal values should be that value."""
        assert geo_mean([4, 4, 4]) == pytest.approx(4.0)
    
    def test_geo_mean_powers_of_two(self):
        """Geometric mean of 1, 4, 16 should be 4."""
        result = geo_mean([1, 4, 16])
        assert result == pytest.approx(4.0, rel=1e-6)
    
    def test_geo_mean_weighted(self):
        """Weighted geometric mean should work correctly."""
        # GM([2, 8], weights=[3, 1]) = 2^(3/4) * 8^(1/4) = 2^(3/4) * 2^(3/4) = 2^(3/2) ≈ 2.83
        result = geo_mean([2, 8], ws=[3, 1])
        assert result == pytest.approx(2.828, rel=1e-2)
    
    def test_geo_mean_less_than_arithmetic(self):
        """Geometric mean should be ≤ arithmetic mean."""
        values = [1, 10, 100]
        gm = geo_mean(values)
        am = np.mean(values)
        assert gm <= am


class TestPctClip:
    """Tests for percentage clipping."""
    
    def test_pct_clip_within_bounds(self):
        """Values within bounds should be unchanged."""
        assert pct_clip(0.5) == 0.5
        assert pct_clip(0.0) == 0.0
        assert pct_clip(-0.5) == -0.5
    
    def test_pct_clip_upper_bound(self):
        """Values above upper bound should be clipped."""
        assert pct_clip(1.5) == 0.99
        assert pct_clip(100.0) == 0.99
    
    def test_pct_clip_lower_bound(self):
        """Values below lower bound should be clipped."""
        assert pct_clip(-1.5) == -0.99
        assert pct_clip(-100.0) == -0.99
    
    def test_pct_clip_custom_bounds(self):
        """Custom bounds should work correctly."""
        assert pct_clip(2.0, lo=-0.5, hi=0.5) == 0.5
        assert pct_clip(-2.0, lo=-0.5, hi=0.5) == -0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
