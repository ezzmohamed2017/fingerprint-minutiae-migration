import numpy as np

from fpmig.preprocess import letterbox, load_grayscale, prepare


def test_letterbox_white_canvas_and_aspect():
    img = np.zeros((500, 250), np.uint8)
    out = letterbox(img)
    assert out.shape == (224, 224) and out.dtype == np.uint8
    assert out[:, :50].min() == 255 and out[:, -50:].min() == 255          # padding is white
    assert out[:, 60:160].max() == 0                                       # the print area


def test_prepare_from_png_bytes():
    import cv2
    img = (np.arange(300 * 200) % 256).astype(np.uint8).reshape(300, 200)
    ok, buf = cv2.imencode(".png", img)
    x = prepare(buf.tobytes())
    assert x.shape == (224, 224, 1)
    np.testing.assert_array_equal(load_grayscale(buf.tobytes()), img)


def test_colour_and_uint16_inputs():
    rgb = np.zeros((100, 120, 3), np.uint8)
    assert load_grayscale(rgb).shape == (100, 120)
    assert load_grayscale(np.full((10, 10), 65535, np.uint16)).max() == 255
