from __future__ import annotations

import numpy as np

from yayoi_ocr.recognize.memory import RecognitionMemory


def _vec(
    seed: int, dim: int = 16, noise: float = 0.0, base: np.ndarray | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim) if base is None else base + noise * rng.normal(size=dim)
    return v / np.linalg.norm(v)


def test_query_returns_nearest_texts():
    mem = RecognitionMemory(k=3)
    a, b = _vec(1), _vec(2)
    for i in range(3):
        mem.add("account", _vec(10 + i, noise=0.05, base=a), "現金")
    mem.add("account", b, "普通預金")
    nb = mem.query("account", _vec(99, noise=0.05, base=a))
    assert [n.text for n in nb] == ["現金", "現金", "現金"] and nb[0].sim > 0.95
    assert mem.query("date", a) == []  # 別の列には無い
    assert mem.size("account") == 4 and len(mem) == 4


def test_votes_weight_by_similarity_and_threshold():
    mem = RecognitionMemory(k=5, sim_threshold=0.9)
    a = _vec(1)
    mem.add("description", _vec(11, noise=0.03, base=a), "切手代")
    mem.add("description", _vec(12, noise=0.03, base=a), "切手代")
    mem.add("description", _vec(13, noise=0.03, base=a), "電話代")
    mem.add("description", _vec(14), "遠い記入")  # 似ていない → 票にならない
    votes, max_sim = mem.votes("description", _vec(50, noise=0.03, base=a))
    assert abs(sum(votes.values()) - 1.0) < 1e-6 and set(votes) <= {"切手代", "電話代"}
    assert votes["切手代"] > votes["電話代"] and max_sim > 0.9
    far_votes, far_sim = mem.votes("description", _vec(77))
    assert far_votes == {} and far_sim < 0.9


def test_column_thresholds_override_and_roundtrip(tmp_path):
    mem = RecognitionMemory(k=3, sim_threshold=0.5, column_thresholds={"amount": 1.1})
    a = _vec(1)
    for col in ("amount", "description"):
        mem.add(col, _vec(11, noise=0.03, base=a), "x")
    q = _vec(50, noise=0.03, base=a)
    assert mem.votes("amount", q)[0] == {}  # 1.1 以上の列は票にならない
    assert mem.votes("amount", q)[1] > 0.9  # 近傍の類似度そのものは返る
    assert mem.votes("description", q)[0] == {"x": 1.0}
    mem.save(tmp_path / "m.npz")
    loaded = RecognitionMemory.load(tmp_path / "m.npz")
    assert loaded.column_thresholds == {"amount": 1.1}
    assert loaded.votes("amount", q)[0] == {}


def test_save_load_roundtrip(tmp_path):
    mem = RecognitionMemory(k=4, sim_threshold=0.8, temperature=0.05)
    for i, t in enumerate(["a", "b", "c"]):
        mem.add("amount", _vec(i), t)
    mem.add("date", _vec(9), "8.21")
    mem.add("date", _vec(10), "")  # 空のテキストは入れない
    p = tmp_path / "m.npz"
    mem.save(p)
    m2 = RecognitionMemory.load(p)
    assert (m2.k, m2.sim_threshold, m2.temperature) == (4, 0.8, 0.05)
    assert m2.size("amount") == 3 and m2.size("date") == 1
    assert m2.query("amount", _vec(1))[0].text == "b"
