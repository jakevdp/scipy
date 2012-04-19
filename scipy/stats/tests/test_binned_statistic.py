#!/usr/bin/env python
import numpy as np
from numpy.testing import \
    assert_array_almost_equal, run_module_suite, TestCase
import numpy.ma as ma
from numpy.ma.testutils import \
    assert_array_almost_equal as assert_array_almost_equal_ma
from scipy.stats import \
    binned_statistic, binned_statistic_2d, binned_statistic_dd


class TestBinnedStatistic(TestCase):
    def test_1d_count(self):
        x = np.random.random(100)
        v = np.random.random(100)

        count1, edges1 = binned_statistic(x, v, 'count', bins=10)
        count2, edges2 = np.histogram(x, bins=10)

        assert_array_almost_equal(count1, count2)
        assert_array_almost_equal(edges1, edges2)


    def test_1d_sum(self):
        x = np.random.random(100)
        v = np.random.random(100)

        sum1, edges1 = binned_statistic(x, v, 'sum', bins=10)
        sum2, edges2 = np.histogram(x, bins=10, weights=v)

        assert_array_almost_equal(sum1, sum2)
        assert_array_almost_equal(edges1, edges2)


    def test_1d_mean(self):
        x = np.random.random(100)
        v = np.random.random(100)

        stat1, edges1 = binned_statistic(x, v, 'mean', bins=10)
        stat2, edges2 = binned_statistic(x, v, np.mean, bins=10)

        assert_array_almost_equal(stat1, stat2)
        assert_array_almost_equal(edges1, edges2)


    def test_1d_median(self):
        x = np.random.random(100)
        v = np.random.random(100)

        stat1, edges1 = binned_statistic(x, v, 'median', bins=10)
        stat2, edges2 = binned_statistic(x, v, np.median, bins=10)

        assert_array_almost_equal(stat1, stat2)
        assert_array_almost_equal(edges1, edges2)


    def test_2d_count(self):
        x = np.random.random(100)
        y = np.random.random(100)
        v = np.random.random(100)

        count1, binx1, biny1 = binned_statistic_2d(x, y, v, 'count', bins=5)
        count2, binx2, biny2 = np.histogram2d(x, y, bins=5)

        assert_array_almost_equal(count1, count2)
        assert_array_almost_equal(binx1, binx2)
        assert_array_almost_equal(biny1, biny2)


    def test_2d_sum(self):
        x = np.random.random(100)
        y = np.random.random(100)
        v = np.random.random(100)

        sum1, binx1, biny1 = binned_statistic_2d(x, y, v, 'sum', bins=5)
        sum2, binx2, biny2 = np.histogram2d(x, y, bins=5, weights=v)

        assert_array_almost_equal(sum1, sum2)
        assert_array_almost_equal(binx1, binx2)
        assert_array_almost_equal(biny1, biny2)


    def test_2d_mean(self):
        x = np.random.random(100)
        y = np.random.random(100)
        v = np.random.random(100)

        stat1, binx1, biny1 = binned_statistic_2d(x, y, v, 'mean', bins=5)
        stat2, binx2, biny2 = binned_statistic_2d(x, y, v, np.mean, bins=5)

        assert_array_almost_equal(stat1, stat2)
        assert_array_almost_equal(binx1, binx2)
        assert_array_almost_equal(biny1, biny2)


    def test_2d_median(self):
        x = np.random.random(100)
        y = np.random.random(100)
        v = np.random.random(100)

        stat1, binx1, biny1 = binned_statistic_2d(x, y, v, 'median', bins=5)
        stat2, binx2, biny2 = binned_statistic_2d(x, y, v, np.median, bins=5)

        assert_array_almost_equal(stat1, stat2)
        assert_array_almost_equal(binx1, binx2)
        assert_array_almost_equal(biny1, biny2)


    def test_dd_count(self):
        X = np.random.random((100, 3))
        v = np.random.random(100)

        count1, edges1 = binned_statistic_dd(X, v, 'count', bins=3)
        count2, edges2 = np.histogramdd(X, bins=3)

        assert_array_almost_equal(count1, count2)
        assert_array_almost_equal(edges1, edges2)


    def test_dd_sum(self):
        X = np.random.random((100, 3))
        v = np.random.random(100)

        sum1, edges1 = binned_statistic_dd(X, v, 'sum', bins=3)
        sum2, edges2 = np.histogramdd(X, bins=3, weights=v)

        assert_array_almost_equal(sum1, sum2)
        assert_array_almost_equal(edges1, edges2)


    def test_dd_mean(self):
        X = np.random.random((100, 3))
        v = np.random.random(100)

        stat1, edges1 = binned_statistic_dd(X, v, 'mean', bins=3)
        stat2, edges2 = binned_statistic_dd(X, v, np.mean, bins=3)

        assert_array_almost_equal(stat1, stat2)
        assert_array_almost_equal(edges1, edges2)


    def test_dd_median(self):
        X = np.random.random((100, 3))
        v = np.random.random(100)

        stat1, edges1 = binned_statistic_dd(X, v, 'median', bins=3)
        stat2, edges2 = binned_statistic_dd(X, v, np.median, bins=3)

        assert_array_almost_equal(stat1, stat2)
        assert_array_almost_equal(edges1, edges2)

    def test_dd_ma_all(self):
        # ``values`` is a masked array with all elements masked
        X = np.tile(np.atleast_2d(np.arange(7) + 0.5), (3, 1)).T
        v = np.arange(7)
        v = ma.masked_all(v.shape)
        bins = [np.arange(0,10, 2), np.arange(0,10, 2), np.arange(0,10, 2)]

        sum1, edges1 = binned_statistic_dd(X, v, 'mean', bins=bins)
        assert_array_almost_equal_ma(sum1, ma.masked_all((4,4,4)))
        sum2, edges2 = binned_statistic_dd(X, v, 'median', bins=bins)
        assert_array_almost_equal_ma(sum2, ma.masked_all((4,4,4)))
        sum3, edges3 = binned_statistic_dd(X, v, 'count', bins=bins)
        assert_array_almost_equal_ma(sum3, ma.masked_all((4,4,4)))
        sum4, edges4 = binned_statistic_dd(X, v, 'sum', bins=bins)
        assert_array_almost_equal_ma(sum4, ma.masked_all((4,4,4)))


    def test_dd_ma_none(self):
        # ``values`` is a masked array with no elements masked
        X = np.tile(np.atleast_2d(np.arange(7) + 0.5), (3, 1)).T
        v = np.arange(7)
        v = ma.masked_array(v)
        bins = [np.arange(0,10, 2), np.arange(0,10, 2), np.arange(0,10, 2)]

        sum1, edges1 = binned_statistic_dd(X, v, 'mean', bins=bins)
        verify1, edges1 = binned_statistic_dd(X, v.data, 'mean', bins=bins)
        assert_array_almost_equal_ma(sum1, verify1)
        sum2, edges2 = binned_statistic_dd(X, v, 'median', bins=bins)
        verify2, edges2 = binned_statistic_dd(X, v.data, 'median', bins=bins)
        assert_array_almost_equal_ma(sum2, verify2)
        sum3, edges3 = binned_statistic_dd(X, v, 'count', bins=bins)
        verify3, edges3 = binned_statistic_dd(X, v.data, 'count', bins=bins)
        assert_array_almost_equal_ma(sum3, verify3)
        sum4, edges4 = binned_statistic_dd(X, v, 'sum', bins=bins)
        verify4, edges4 = binned_statistic_dd(X, v.data, 'sum', bins=bins)
        assert_array_almost_equal_ma(sum4, verify4)

    def test_dd_ma(self):
        # ``values`` is a masked array with some elements masked
        X = np.tile(np.atleast_2d(np.arange(7) + 0.5), (3, 1)).T
        v = np.arange(7)
        v = ma.masked_array(v)
        v[2], v[5] = ma.masked, ma.masked
        bins = [np.arange(0,10, 2), np.arange(0,10, 2), np.arange(0,10, 2)]

        sum1, edges1 = binned_statistic_dd(X, v, 'mean', bins=bins)
        verify1 = np.ones((4,4,4)) * np.nan
        verify1[0,0,0] = 0.5
        verify1[1,1,1] = 3
        verify1[2,2,2] = 4
        verify1[3,3,3] = 6
        assert_array_almost_equal_ma(sum1, verify1)
        sum2, edges2 = binned_statistic_dd(X, v, 'median', bins=bins)
        verify2 = np.ones((4,4,4)) * np.nan
        verify2[0,0,0] = 0.5
        verify2[1,1,1] = 3
        verify2[2,2,2] = 4
        verify2[3,3,3] = 6
        assert_array_almost_equal_ma(sum2, verify2)
        sum3, edges3 = binned_statistic_dd(X, v, 'count', bins=bins)
        verify3 = np.zeros((4,4,4))
        verify3[0,0,0] = 2
        verify3[1,1,1] = 2
        verify3[2,2,2] = 2
        verify3[3,3,3] = 1
        assert_array_almost_equal_ma(sum3, verify3)
        sum4, edges4 = binned_statistic_dd(X, v, 'sum', bins=bins)
        verify4 = np.zeros((4,4,4))
        verify4[0,0,0] = 1
        verify4[1,1,1] = 3
        verify4[2,2,2] = 4
        verify4[3,3,3] = 6
        assert_array_almost_equal_ma(sum4, verify4)
        sum5, edges5 = binned_statistic_dd(X, v, 'mean', bins=bins, mask_invalid=True)
        verify5 = np.zeros((4,4,4))
        verify5[0,0,0] = 0.5
        verify5[1,1,1] = 3
        verify5[2,2,2] = 4
        verify5[3,3,3] = 6
        assert_array_almost_equal_ma(sum5, verify5)
        sum6, edges6 = binned_statistic_dd(X, v, 'median', bins=bins, mask_invalid=True)
        verify6 = np.zeros((4,4,4))
        verify6[0,0,0] = 0.5
        verify6[1,1,1] = 3
        verify6[2,2,2] = 4
        verify6[3,3,3] = 6
        assert_array_almost_equal_ma(sum6, verify6)

if __name__ == "__main__":
    run_module_suite()
