"""見覚えメモリ(2.4・推論側): 確定済みのセルの画像埋め込みと確定テキストを列ごとに持ち、kNN で候補を足す。

- 保存するもの: 確定したセルごとに、R1 のエンコーダ出力を横方向 4 ビンに分けて平均・連結し L2 正規化した埋め込み(192 x 4 次元)と確定テキスト(書かれた通りの文字列)。
- 推論時: 新しいセルの埋め込みに近い(コサイン類似度)上位 k 件の確定テキストを、類似度で重みづけして投票させる。
  票を集めたテキストは R1 の `score()` で採点し直す候補に加え、事後確率に事前(票の割合)として反映する(`fields.py`)。
- 類似度のしきい値・温度は合成の書き手で較正する(`scripts/sim_learning.py`)。
- 純 numpy。保存は npz + JSON(pickle を使わない)。学習(Phase 4 の追加学習)とは別の、すぐ効く記憶。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Neighbor:
    text: str
    sim: float


class RecognitionMemory:
    def __init__(self, k: int = 5, sim_threshold: float = 0.85, temperature: float = 0.04) -> None:
        self.k = k
        self.sim_threshold = sim_threshold  # これ未満の類似度の近傍は票にしない
        self.temperature = temperature  # 票の重み exp((sim - 1) / T)
        self._emb: dict[str, list[np.ndarray]] = defaultdict(list)
        self._txt: dict[str, list[str]] = defaultdict(list)
        self._mat: dict[str, np.ndarray | None] = {}  # 列ごとの行列キャッシュ

    def __len__(self) -> int:
        return sum(len(v) for v in self._txt.values())

    def size(self, column: str) -> int:
        return len(self._txt.get(column, []))

    def add(self, column: str, embedding: np.ndarray, text: str) -> None:
        if not text:
            return
        e = np.asarray(embedding, dtype=np.float32)
        n = float(np.linalg.norm(e))
        if n < 1e-8:
            return
        self._emb[column].append(e / n)
        self._txt[column].append(text)
        self._mat[column] = None

    def _matrix(self, column: str) -> np.ndarray | None:
        if column not in self._txt or not self._txt[column]:
            return None
        if self._mat.get(column) is None:
            self._mat[column] = np.stack(self._emb[column])
        return self._mat[column]

    def query(self, column: str, embedding: np.ndarray) -> list[Neighbor]:
        """類似度の高い順に最大 k 件(しきい値に関係なく)。"""
        m = self._matrix(column)
        if m is None:
            return []
        e = np.asarray(embedding, dtype=np.float32)
        e = e / max(float(np.linalg.norm(e)), 1e-8)
        sims = m @ e
        k = min(self.k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [Neighbor(self._txt[column][i], float(sims[i])) for i in idx]

    def votes(self, column: str, embedding: np.ndarray) -> tuple[dict[str, float], float]:
        """テキスト → 票の割合(合計 1。しきい値以上の近傍だけ)と、近傍の最大類似度。"""
        nb = self.query(column, embedding)
        max_sim = nb[0].sim if nb else 0.0
        w: dict[str, float] = defaultdict(float)
        for n in nb:
            if n.sim >= self.sim_threshold:
                w[n.text] += float(np.exp((n.sim - 1.0) / self.temperature))
        total = sum(w.values())
        if total <= 0:
            return {}, max_sim
        return {t: v / total for t, v in w.items()}, max_sim

    # ---- 保存・読み込み(npz + JSON。pickle なし)
    def save(self, path: str | Path) -> None:
        arrays = {f"emb_{c}": np.stack(v) for c, v in self._emb.items() if v}
        meta = json.dumps(
            {
                "k": self.k,
                "sim_threshold": self.sim_threshold,
                "temperature": self.temperature,
                "texts": {c: t for c, t in self._txt.items()},
            },
            ensure_ascii=False,
        )
        np.savez_compressed(Path(path), meta=np.array(meta), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> RecognitionMemory:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            mem = cls(meta["k"], meta["sim_threshold"], meta["temperature"])
            for c, texts in meta["texts"].items():
                if not texts:
                    continue
                emb = z[f"emb_{c}"]
                mem._emb[c] = [row for row in emb]
                mem._txt[c] = list(texts)
        return mem
