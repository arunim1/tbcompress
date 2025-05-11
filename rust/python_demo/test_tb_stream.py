import tb_stream, numpy as np


def test_shape_and_wdl():
    it = tb_stream.make_tb_stream("../Syzygy345_WDL", seed=1234)
    for _ in range(1000):
        planes, wdl = next(it)
        assert planes.shape == (12, 8, 8)
        assert planes.dtype == np.uint8
        assert wdl in (-1, 0, 1)
