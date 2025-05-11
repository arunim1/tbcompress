import tb_stream, numpy as np


def test_shape_and_wdl():
    try:
        it = tb_stream.make_tb_stream("./Syzygy345_WDL", seed=1234, batch_size=128)
        count = 0
        for planes, wdl in it:
            assert planes.shape == (128, 12, 8, 8)
            assert planes.dtype == np.uint8
            assert wdl.shape == (128,)
            assert wdl.dtype == np.int8
            assert wdl.min() in (-1, 0, 1)
            count += 1
            if count >= 1000:
                break
        print("✅ test_shape_and_wdl")
    except Exception as e:
        print("❌ test_shape_and_wdl")
        raise e


test_shape_and_wdl()
