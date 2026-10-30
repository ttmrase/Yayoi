"""PARSeq(NDLOCR-Lite ver1.2)の自作実装と、同梱 ONNX との数値一致テスト。

ライセンス・出所の根拠(「取り込んだモデルは NDLOCR-Lite 同梱のものと同じ」)を示すテストなので、
必要なファイル(PyTorch・onnxruntime・onnx と、`scripts/fetch_models.py` が取得する ONNX)が無い環境では skip する。
テスト入力は合成文字列(OFL フォント)と乱数画像だけ。実データは使わない。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from yayoi_ocr.config import models_dir

pytestmark = [pytest.mark.ml, pytest.mark.slow]

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from yayoi_ocr.recognize.parseq.charset import load_charset  # noqa: E402
from yayoi_ocr.recognize.parseq.preprocess import ndl_preprocess  # noqa: E402
from yayoi_ocr.recognize.parseq.weights import (  # noqa: E402
    load_converted,
    load_parseq,
    load_parseq_onnx,
    state_dict_from_onnx,
)

STEMS = {
    256: "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604",
    384: "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604",
}
PAPER = (245, 245, 245)  # allow-int-list
INK = (30, 30, 30)  # allow-int-list
LOGIT_TOL = 2e-3  # 浮動小数点の演算順序の違い程度(実測 256: 2.4e-4, 384: 7e-4)


def _need(*names: str):
    for n in names:
        if not (models_dir() / n).is_file():
            pytest.skip(f"{models_dir() / n} がありません(scripts/fetch_models.py)")


def _images(n_noise: int = 6) -> list[np.ndarray]:
    """合成文字列(取得済みの OFL フォントがあれば)と乱数画像。"""
    from PIL import Image, ImageDraw, ImageFont

    from yayoi_ocr.synth.fonts import find_handwriting_fonts

    rng = np.random.default_rng(0)
    imgs: list[np.ndarray] = []
    texts = ["5.12", "R7 11.19", "3,456-", "現金", "切手代", "電話代 8月分", "〃", "旅費交通費"]
    for font in find_handwriting_fonts()[:3]:
        f = ImageFont.truetype(str(font), 48)
        for t in texts:
            left, top, right, bottom = f.getbbox(t)
            im = Image.new("RGB", (right - left + 20, bottom - top + 20), PAPER)
            ImageDraw.Draw(im).text((10 - left, 10 - top), t, fill=INK, font=f)
            imgs.append(np.asarray(im)[:, :, ::-1].copy())
    for _ in range(n_noise):
        shape = (40, int(rng.integers(40, 300)), 3)
        imgs.append((rng.random(shape) * 255).astype(np.uint8))
    return imgs


@pytest.mark.parametrize("width", [256, 384])
def test_matches_bundled_onnx(width):
    """ONNX の重みで動かした自作モデルの出力(logits)が、ONNX Runtime の出力と数値的に一致する。"""
    onnx_path = models_dir() / f"{STEMS[width]}.onnx"
    _need(f"{STEMS[width]}.onnx", "NDLmoji.yaml")
    charset = load_charset(models_dir() / "NDLmoji.yaml")
    model = load_parseq_onnx(onnx_path, width, charset)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for im in _images():
            x = ndl_preprocess(im, width, 24, rotate_tall=True)  # NDLOCR-Lite と同じ前処理
            ref = sess.run(None, {"images": x})[0]
            with torch.inference_mode():
                out = model(torch.from_numpy(x)).numpy()
            assert out.shape == ref.shape
            worst = max(worst, float(np.abs(out - ref).max()))
            assert (out.argmax(-1) == ref.argmax(-1)).all()
    assert worst < LOGIT_TOL


def test_cloze_mask_variant_differs_from_onnx():
    """baudm/parseq の PyTorch 実装(cloze で自分自身を遮蔽)は ONNX と挙動が違う(ONNX の癖の記録)。"""
    width = 256
    _need(f"{STEMS[width]}.onnx", "NDLmoji.yaml")
    charset = load_charset(models_dir() / "NDLmoji.yaml")
    model = load_parseq_onnx(models_dir() / f"{STEMS[width]}.onnx", width, charset)
    model.refine_mask = "cloze"
    sess = ort.InferenceSession(
        str(models_dir() / f"{STEMS[width]}.onnx"), providers=["CPUExecutionProvider"]
    )
    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for im in _images()[:8]:
            x = ndl_preprocess(im, width, 24, rotate_tall=True)
            ref = sess.run(None, {"images": x})[0]
            with torch.inference_mode():
                worst = max(worst, float(np.abs(model(torch.from_numpy(x)).numpy() - ref).max()))
    assert worst > 0.1  # ONNX のマスクは -inf でなく +1.0 の加算バイアス


@pytest.mark.parametrize("width", [256, 384])
def test_converted_weights_roundtrip(width):
    """fetch_models.py が作った .pt が、ONNX から復元した重みと同一。"""
    stem = STEMS[width]
    pt = {256: "parseq-ndl-24x256-30-tiny.pt", 384: "parseq-ndl-24x384-50-tiny.pt"}[width]
    _need(pt, f"{stem}.onnx")
    model, charset, source = load_converted(models_dir() / pt)
    assert len(charset) == 7141
    assert source["commit"]
    ref = state_dict_from_onnx(models_dir() / f"{stem}.onnx")
    for k, v in model.state_dict().items():
        assert torch.equal(v, ref[k]), k


def test_ckpt_vs_onnx_weights_report():
    """NDL 提供の ckpt と同梱 ONNX の重みの関係(参考・記録): 384 は同一、256 は別スナップショット。"""
    _need(f"{STEMS[384]}.ckpt", f"{STEMS[384]}.onnx", f"{STEMS[256]}.ckpt", f"{STEMS[256]}.onnx")
    rel = {}
    for w in (256, 384):
        model, _ = load_parseq(models_dir() / f"{STEMS[w]}.ckpt")
        ref = state_dict_from_onnx(models_dir() / f"{STEMS[w]}.onnx")
        rel[w] = max(
            float((v - ref[k]).norm() / (ref[k].norm() + 1e-12))
            for k, v in model.state_dict().items()
        )
    assert rel[384] == 0.0  # ckpt(epoch 299)と ONNX がビット同一
    assert rel[256] > 1e-2  # ckpt(epoch 253)と ONNX(epoch 189 のファイル名)は別物
