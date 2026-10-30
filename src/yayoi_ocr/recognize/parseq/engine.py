"""PARSeq(NDLOCR-Lite ver1.2, 高さ 24)による文字列認識エンジン。

- `recognize(crop, column)`: 候補(文字列・系列の対数尤度・文字ごとの確信度)を返す。貪欲 + 反復修正 1 回
  (NDLOCR-Lite の ONNX と同じ計算)の結果を先頭に、ビームサーチ(既定 5)の候補を尤度の降順で続ける。
- `score(crop, text)`: 与えた文字列の教師強制での対数尤度(辞書・過去の確定値との照合に使う)。
- 列ごとの文字集合マスク(日付 0-9 R . 〃 空白 / 金額 0-9 , -)を logits に掛ける(EOS は常に許可)。
- 認識後に全角の数字・英字を半角にする。
- 重み(ckpt)はリポジトリに含めない。`$YAYOI_PRIVATE_DATA/models/` から読む(`scripts/fetch_models.py`)。
  実行時にダウンロードしない(オフライン前提)。

幅は短い欄(日付・科目・金額)が 256 px(最大 30 字)、摘要が 384 px(最大 50 字)。
入力は「インクの外接矩形 + 余白」で切った画像(`crops.recognition_image` が作る)を想定する。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ...config import models_dir as default_models_dir
from ...models import ColumnKind
from .charset import column_charset, normalize_width
from .model import PARSeq
from .preprocess import ndl_preprocess
from .weights import load_converted

# scripts/fetch_models.py が NDLOCR-Lite 同梱の ONNX から作る重み(テンソルのみの .pt)
MODEL_FILES: dict[int, str] = {
    256: "parseq-ndl-24x256-30-tiny.pt",
    384: "parseq-ndl-24x384-50-tiny.pt",
}
HEIGHT = 24
EMBED_BINS = 4  # 見覚えメモリの埋め込みで、横方向を何個に分けて平均するか(1 = 全体の平均)
# 推論モード(設定): onnx = NDLOCR-Lite 配布版と同じ反復修正 / cloze = baudm/parseq の反復修正 / none = 自己回帰のみ。
# Phase 4 の追加学習の後も、ホールドアウトで再評価して選べるように設定として持つ。
REFINE_MODES = ("onnx", "cloze", "none")


@dataclass
class ParseqResult:
    """1 つの候補。`logprob` は教師強制(AR)での文字 + 終端 EOS の対数尤度。"""

    text: str  # 全角英数を半角にした認識文字列
    logprob: float  # 系列の対数尤度(-inf = 許可されない文字を含む)
    char_confidences: list[float]  # 各文字(トークン)の確率。len == n_tokens
    eos_prob: float  # 「ここで終わる」確率(終端 EOS のトークン確率)
    source: str  # "greedy" | "beam" | "score"
    tokens: tuple[int, ...] = field(default_factory=tuple, repr=False)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def mean_logprob(self) -> float:
        """トークン(文字 + EOS)あたりの平均対数尤度。長さの違う候補の比較用。"""
        return self.logprob / (self.n_tokens + 1)

    @property
    def min_char_confidence(self) -> float:
        return min(self.char_confidences, default=1.0)


@dataclass
class RecognizeOutput:
    candidates: list[ParseqResult]
    width: int
    embedding: np.ndarray | None = (
        None  # エンコーダ出力の横方向ビン平均 → L2 正規化(見覚えメモリ用)
    )
    memory: torch.Tensor | None = field(default=None, repr=False)  # (1, T, C) 再採点用に保持

    @property
    def best(self) -> ParseqResult:
        return self.candidates[0]


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParseqEngine:
    """R1。ckpt は遅延読み込み(幅ごと)。スレッドセーフではない(1 スレッドから使う)。"""

    def __init__(
        self,
        models_dir: str | Path | None = None,
        device: str | None = None,
        widths: Sequence[int] = (256, 384),
        refine_mode: str = "onnx",
    ) -> None:
        if refine_mode not in REFINE_MODES:
            raise ValueError(f"refine_mode は {REFINE_MODES} のいずれか: {refine_mode}")
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self.device = pick_device(device)
        self.widths = tuple(widths)
        self.refine_mode = refine_mode
        self._models: dict[int, PARSeq] = {}
        self._charset: str | None = None
        self._tok: dict[str, int] = {}
        self._mask_cache: dict[object, torch.Tensor] = {}

    @classmethod
    def from_models(
        cls, models: dict[int, PARSeq], charset: str, device: str | None = None
    ) -> ParseqEngine:
        """読み込み済みのモデル(幅 → PARSeq)から作る(テスト・学習中のモデルの評価用)。"""
        eng = cls(widths=tuple(models), device=device)
        eng._charset = charset
        eng._tok = {ch: i + 1 for i, ch in enumerate(charset)}
        eng._models = {w: m.eval().to(eng.device) for w, m in models.items()}
        return eng

    # ------------------------------------------------------------------ 読み込み

    def available(self) -> bool:
        return all((self.models_dir / MODEL_FILES[w]).is_file() for w in self.widths)

    def _model(self, width: int) -> PARSeq:
        if width not in self._models:
            path = self.models_dir / MODEL_FILES[width]
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} がありません。scripts/fetch_models.py で取得してください(開発用)。"
                )
            model, charset, _ = load_converted(path)
            if self._charset is None:
                self._charset = charset
                self._tok = {ch: i + 1 for i, ch in enumerate(charset)}
            elif charset != self._charset:
                raise ValueError("幅の違う ckpt で文字集合が一致しません")
            self._models[width] = model.to(self.device)
        return self._models[width]

    @property
    def charset(self) -> str:
        self._model(self.widths[0])
        assert self._charset is not None
        return self._charset

    @staticmethod
    def default_width(column: ColumnKind | str | None) -> int:
        return 384 if column in (ColumnKind.DESCRIPTION, "description") else 256

    # ------------------------------------------------------------------ マスク・トークン

    def vocab_mask(self, column: ColumnKind | str | None) -> torch.Tensor | None:
        """列の許可文字から作る (V,) bool(EOS は常に許可)。制限なしなら None。"""
        allowed = column_charset(column)
        if allowed is None:
            return None
        key = tuple(sorted(allowed))
        if key not in self._mask_cache:
            self.charset  # noqa: B018  (読み込みを保証)
            m = torch.zeros(len(self._charset or "") + 1, dtype=torch.bool)
            m[0] = True
            for ch in allowed:
                if ch in self._tok:
                    m[self._tok[ch]] = True
            self._mask_cache[key] = m
        return self._mask_cache[key].to(self.device)

    def encode_text(self, text: str) -> list[int] | None:
        """文字列 → トークン ID。文字集合にない文字を含めば None。"""
        self.charset  # noqa: B018
        ids = []
        for ch in normalize_width(text):
            t = self._tok.get(ch)
            if t is None:
                return None
            ids.append(t)
        return ids

    def decode_tokens(self, ids: Sequence[int]) -> str:
        cs = self.charset
        out = []
        for t in ids:
            if t == 0:
                break
            if 1 <= t <= len(cs):
                out.append(cs[t - 1])
        return "".join(out)

    # ------------------------------------------------------------------ 前処理・符号化

    def _batch(self, crops: Sequence[np.ndarray], width: int) -> torch.Tensor:
        x = np.concatenate([ndl_preprocess(c, width, HEIGHT) for c in crops], axis=0)
        return torch.from_numpy(x).to(self.device)

    def _row_mask(self, columns: Sequence[ColumnKind | str | None]) -> torch.Tensor | None:
        masks = [self.vocab_mask(c) for c in columns]
        if all(m is None for m in masks):
            return None
        full = torch.ones(len(self.charset) + 1, dtype=torch.bool, device=self.device)
        return torch.stack([full if m is None else m for m in masks])

    @staticmethod
    def _embed(model: PARSeq, memory: torch.Tensor) -> np.ndarray:
        """エンコーダ出力 (B, 行 x 列, C) → 縦方向に平均 → 横方向を EMBED_BINS 個に分けて平均 → 連結 → L2 正規化。

        全体の平均より、文字の並び(横方向の位置)を残せるので、同じ筆者が同じ語を書いたときの照合に強い。"""
        gh, gw = model.encoder.patch_embed.grid_size
        b, _, c = memory.shape
        cols = memory.reshape(b, gh, gw, c).mean(dim=1)  # (B, gw, C)
        e = torch.nn.functional.adaptive_avg_pool1d(
            cols.transpose(1, 2), EMBED_BINS
        )  # (B, C, bins)
        e = e.reshape(b, -1)
        e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return e.float().cpu().numpy()

    # ------------------------------------------------------------------ 認識

    @torch.inference_mode()
    def recognize_many(
        self,
        crops: Sequence[np.ndarray],
        columns: Sequence[ColumnKind | str | None],
        *,
        beam: int = 5,
        width: int | None = None,
        with_embedding: bool = False,
        batch_size: int = 32,
        keep_memory: bool = False,
    ) -> list[RecognizeOutput]:
        """複数のセル画像を認識する(幅・列が違ってもよい。幅ごとにまとめてバッチ処理)。"""
        if len(crops) != len(columns):
            raise ValueError("crops と columns の数が違います")
        widths = [width or self.default_width(c) for c in columns]
        results: list[RecognizeOutput | None] = [None] * len(crops)
        for w in sorted(set(widths)):
            idx = [i for i, ww in enumerate(widths) if ww == w]
            model = self._model(w)
            for s in range(0, len(idx), batch_size):
                sub = idx[s : s + batch_size]
                outs = self._recognize_batch(
                    model,
                    w,
                    [crops[i] for i in sub],
                    [columns[i] for i in sub],
                    beam,
                    with_embedding,
                    keep_memory,
                )
                for i, o in zip(sub, outs, strict=True):
                    results[i] = o
        return [r for r in results if r is not None]

    def recognize(
        self,
        crop: np.ndarray,
        column: ColumnKind | str | None,
        *,
        beam: int = 5,
        width: int | None = None,
        with_embedding: bool = False,
        keep_memory: bool = False,
    ) -> RecognizeOutput:
        return self.recognize_many(
            [crop],
            [column],
            beam=beam,
            width=width,
            with_embedding=with_embedding,
            keep_memory=keep_memory,
        )[0]

    def _recognize_batch(
        self,
        model: PARSeq,
        width: int,
        crops: Sequence[np.ndarray],
        columns: Sequence[ColumnKind | str | None],
        beam: int,
        with_embedding: bool,
        keep_memory: bool = False,
    ) -> list[RecognizeOutput]:
        x = self._batch(crops, width)
        vmask = self._row_mask(columns)
        memory = model.encode(x)
        n = len(crops)
        # 貪欲 + 反復修正(ONNX と同じ計算)
        model.refine_mask = "cloze" if self.refine_mode == "cloze" else "onnx"
        logits = model(x, vocab_mask=vmask, refine=self.refine_mode != "none")
        limit = model.max_label_length  # EOS が出ないまま最大長に達したら、文字数の上限で切る
        greedy_ids = [self._ids_until_eos(row)[:limit] for row in logits.argmax(-1).tolist()]
        cand_ids: list[list[tuple[list[int], str]]] = [[(g, "greedy")] for g in greedy_ids]
        if beam > 1:
            toks, bscores = model.beam_search(memory, beam, vmask)
            for b in range(n):
                seen = {tuple(greedy_ids[b])}
                for k in range(toks.shape[1]):
                    if not math.isfinite(float(bscores[b, k])):
                        continue
                    ids = self._ids_until_eos(toks[b, k].tolist())[:limit]
                    if tuple(ids) not in seen:
                        seen.add(tuple(ids))
                        cand_ids[b].append((ids, "beam"))
        # すべての候補を同じ定義(教師強制の AR)で採点し直す
        flat = [(b, ids, src) for b in range(n) for ids, src in cand_ids[b]]
        rows = [b for b, _, _ in flat]
        kv_rep = [
            (k[rows], v[rows]) for k, v in model.project_memory(memory)
        ]  # K・V は 1 回だけ射影
        mask_rep = None if vmask is None else vmask[rows]
        picked, valid = model.token_logprobs(kv_rep, [ids for _, ids, _ in flat], mask_rep)
        picked = picked.cpu()
        valid = valid.cpu()
        emb = self._embed(model, memory) if with_embedding else None
        outs: list[RecognizeOutput] = []
        pos = 0
        for b in range(n):
            cands = []
            for ids, src in cand_ids[b]:
                cands.append(self._make_result(ids, picked[pos], valid[pos], src))
                pos += 1
            head, tail = cands[:1], sorted(cands[1:], key=lambda r: -r.logprob)
            outs.append(
                RecognizeOutput(
                    head + tail,
                    width,
                    None if emb is None else emb[b],
                    memory[b : b + 1].clone() if keep_memory else None,
                )
            )
        return outs

    @staticmethod
    def _ids_until_eos(ids: Sequence[int]) -> list[int]:
        out = []
        for t in ids:
            if t == 0:
                break
            out.append(int(t))
        return out

    def _make_result(
        self, ids: list[int], picked: torch.Tensor, valid: torch.Tensor, source: str
    ) -> ParseqResult:
        lp = picked[: len(ids) + 1].tolist()
        text = normalize_width(self.decode_tokens(ids))
        total = float(sum(lp)) if all(math.isfinite(v) for v in lp) else float("-inf")
        conf = [math.exp(v) for v in lp[: len(ids)]]
        return ParseqResult(
            text=text,
            logprob=total,
            char_confidences=conf,
            eos_prob=math.exp(lp[len(ids)]),
            source=source,
            tokens=tuple(ids),
        )

    # ------------------------------------------------------------------ 採点

    @torch.inference_mode()
    def score_many(
        self,
        crop: np.ndarray,
        texts: Sequence[str],
        column: ColumnKind | str | None = None,
        *,
        width: int | None = None,
    ) -> list[ParseqResult]:
        """1 枚のセル画像に対して、各文字列の教師強制の対数尤度を返す。

        文字集合にない文字を含む文字列は logprob = -inf。列を与えると、その列の許可文字のマスクを掛ける
        (許可されない文字を含む文字列は -inf)。
        """
        w = width or self.default_width(column)
        model = self._model(w)
        memory = model.encode(self._batch([crop], w))
        return self._score_memory(model, memory, texts, column)

    def _score_memory(
        self,
        model: PARSeq,
        memory: torch.Tensor,
        texts: Sequence[str],
        column: ColumnKind | str | None,
    ) -> list[ParseqResult]:
        enc = [self.encode_text(t) for t in texts]
        ok = [i for i, e in enumerate(enc) if e is not None and len(e) <= model.max_label_length]
        outs: list[ParseqResult] = [
            ParseqResult(t, float("-inf"), [], 0.0, "score", ()) for t in texts
        ]
        if ok:
            vm = self.vocab_mask(column)
            ids = [enc[i] for i in ok]  # type: ignore[misc]
            picked, valid = model.token_logprobs(memory, ids, vm)  # メモリ(バッチ 1)は全候補で共有
            picked, valid = picked.cpu(), valid.cpu()
            for j, i in enumerate(ok):
                outs[i] = self._make_result(list(ids[j]), picked[j], valid[j], "score")
                outs[i].text = texts[i]
        return outs

    @torch.inference_mode()
    def score_output(
        self,
        out: RecognizeOutput,
        texts: Sequence[str],
        column: ColumnKind | str | None = None,
    ) -> list[ParseqResult]:
        """`recognize(..., keep_memory=True)` の結果に対して、各文字列の教師強制の対数尤度を返す(再エンコードしない)。"""
        if out.memory is None:
            raise ValueError("keep_memory=True で認識した結果が必要です")
        model = self._model(out.width)
        return self._score_memory(model, out.memory, texts, column)

    def score(
        self,
        crop: np.ndarray,
        text: str,
        column: ColumnKind | str | None = None,
        *,
        width: int | None = None,
    ) -> ParseqResult:
        return self.score_many(crop, [text], column, width=width)[0]

    # ------------------------------------------------------------------ 埋め込み

    @torch.inference_mode()
    def embed_many(
        self, crops: Sequence[np.ndarray], width: int = 256, batch_size: int = 64
    ) -> np.ndarray:
        """見覚えメモリ用の埋め込み (N, 192 x EMBED_BINS)(`_embed`)。"""
        model = self._model(width)
        outs = []
        for s in range(0, len(crops), batch_size):
            outs.append(
                self._embed(model, model.encode(self._batch(crops[s : s + batch_size], width)))
            )
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 192 * EMBED_BINS), np.float32)
