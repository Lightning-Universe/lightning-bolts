from math import sqrt

from numpy.testing import assert_almost_equal

from pl_bolts.models.rl.common.running_statistics import RunningStatistics, ZFilter


def test_running_statistics():
    rs = RunningStatistics(shape=2)
    rs.push([1, 10])

    assert list(rs.mean) == [1, 10]

    rs.push([1, 20])
    assert list(rs.mean) == [1, 15]
    assert list(rs.var) == [0, 50]
    assert_almost_equal(list(rs.std), [0, sqrt(50)], decimal=3)

    rs.push([4, 15])
    assert list(rs.mean) == [2, 15]
    assert list(rs.var) == [3, 25]
    assert_almost_equal(list(rs.std), [sqrt(3), 5], decimal=3)


def test_z_filter():
    rs = ZFilter(shape=2)

    normalized_x = rs([1, 10])
    normalized_x = rs([1, 20])
    assert_almost_equal(list(normalized_x), [0, 0.7071067801865475], decimal=3)

    normalized_x = rs([4, 15])
    assert_almost_equal(list(normalized_x), [2, 15], decimal=3)
