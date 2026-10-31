"""追加学習の別プロセス(`python -I -m yayoi_ocr.app.train_worker <作業ディレクトリ>`)。設計は `docs/phase4_design.md`。

作業ディレクトリの `plan.json`(`training.TrainManager._plan` が作る)を読み、次を行って `result.json` に書く(進み具合は `progress.json`):

1. **ラベル**: 学習用ページの確定セル(種別が text)から、確定値と整合する書き方のうち、出荷時の重みで最も尤度の高いものを正解にする(`recognize/labels.py`)。
2. **学習**: 出発点は常に**出荷時の重み**(世代を重ねて累積せず、学習用ページ全体からやり直す。L2-SP の基準も出荷時の重み)。後段 4 ブロック + デコーダ + 出力層、L2-SP 0.1、学習率 3e-4、
   強い拡張、バッチ 32。幅 256 = 日付・科目・金額、幅 384 = 摘要。利用者のセル数の 50% の合成セル(`rehearsal.npz`)を混ぜる(忘れ防止)。のべ 4,000 サンプルで頭打ち。
   CPU のときはスレッド数を制限し優先度を下げる。GPU があれば使う(**GPU の経路は試用版では未検証**)。
3. **評価**: 評価用ページを**親(使用中の世代)**と**新しい世代**の両方で認識し直し(文脈は学習用ページから作った辞書・金額履歴・設定の科目)、確定値と比べた件数を返す。
4. **再較正**: 新しい世代の確信度は、全体の較正から出発し、観測した見逃しが目標を超えた列だけしきい値を上げる(`recalibrate.py`)。変えた列があれば、もう一度評価する。

標準出力(= `train.log`)には**ステージと件数だけ**を出す(摘要・金額などの値や、例外の文面は出さない)。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import evalstats, gate, recalibrate
from .trainers import WEIGHTS_SUBDIR

Progress = Callable[[str, float], None]

REHEARSAL_FRACTION = 0.5
MAX_PASSES = max(
    1, min(4000, int(os.environ.get("YAYOI_TRAIN_MAX_PASSES", "4000")))
)  # 1 回の追加学習で見るサンプル数(のべ)の上限(環境変数は試験用。下げる方向にだけ効く)
LR = 3e-4
L2SP = 0.1
BATCH = 32
FROZEN_BLOCKS = 8


class WorkerError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def pick_device(plan: dict, torch) -> str:
    """計画の `device`("auto" / "cpu" / "cuda")→ 実際のデバイス。GPU が無ければ CPU。"""
    want = plan.get("device", "auto")
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def epochs_for(n_samples: int) -> int:
    """のべ `MAX_PASSES` サンプルで頭打ち: エポック数 = clamp(round(4000 / サンプル数), 1, 40)。"""
    return int(np.clip(round(MAX_PASSES / max(n_samples, 1)), 1, 40))


def mix_rehearsal(
    user: list[tuple[np.ndarray, str]],
    pool: list[tuple[np.ndarray, str]],
    rng: np.random.Generator,
    fraction: float = REHEARSAL_FRACTION,
) -> tuple[list[tuple[np.ndarray, str]], int]:
    """利用者のセルに、その `fraction` の合成セルを混ぜる。のべ `MAX_PASSES` を超えるときは無作為に間引く。(混ぜた後のサンプル, 混ぜた件数)。"""
    k = min(len(pool), int(fraction * len(user)))
    chosen = [pool[int(i)] for i in rng.choice(len(pool), k, replace=False)] if k else []
    samples = user + chosen
    if len(samples) > MAX_PASSES:
        keep = rng.choice(len(samples), MAX_PASSES, replace=False)
        samples = [samples[int(i)] for i in sorted(keep)]
    return samples, k


def run(plan: dict, run_dir: Path, progress: Progress | None = None) -> dict:
    """学習・評価・再較正を行い、結果(`result.json` の中身)を返す。失敗は WorkerError。"""
    import torch

    from ..config import models_dir
    from ..recognize import variants
    from ..recognize.celltype import CellTypeClassifier
    from ..recognize.confidence import HARD_REASONS, ConfidenceModel
    from ..recognize.labels import WIDTH_COLUMNS, choose_label
    from ..recognize.page import JobContext, PageRecognizer
    from ..recognize.pagedata import load_page_input
    from ..recognize.parseq.engine import MODEL_FILES, ParseqEngine
    from ..recognize.parseq.finetune import FinetuneConfig, finetune
    from ..recognize.parseq.weights import load_converted, save_converted
    from ..recognize.rehearsal import FILE as REHEARSAL_FILE
    from ..recognize.rehearsal import load_rehearsal
    from ..recognize.state import ConfirmedCell

    def prog(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, frac)

    t_start = time.time()
    run_id = int(plan.get("run_id", 0))
    seed = run_id % 1000
    device = pick_device(plan, torch)
    threads = int(plan.get("threads", 2))
    if device == "cpu":
        torch.set_num_threads(
            max(1, threads)
        )  # CPU のときはスレッド数を制限する(他の計算と競合させない)
    prog("prepare", 0.0)
    shipped = variants.default_variant()
    init_dir = variants.variant_dir(shipped)
    parent = plan["parent"]
    parent_dir = Path(parent["weights_dir"]) if parent.get("weights_dir") else init_dir
    parent_conf = (
        Path(parent["confidence"])
        if parent.get("confidence")
        else variants.confidence_path(shipped)
    )
    global_conf = variants.confidence_path(shipped)
    if not global_conf.is_file():
        raise WorkerError("confidence_missing")
    clf_path = models_dir() / "celltype.pt"
    clf = CellTypeClassifier.load(clf_path) if clf_path.is_file() else None
    rehearsal_path = models_dir() / REHEARSAL_FILE
    rehearsal = None
    if rehearsal_path.is_file():
        rehearsal = load_rehearsal(rehearsal_path)
    elif not plan.get("allow_no_rehearsal"):
        raise WorkerError("rehearsal_missing")

    def load_init() -> tuple[dict, str, dict]:
        models, sources, charset = {}, {}, ""
        for w in (256, 384):
            m, charset, src = load_converted(init_dir / MODEL_FILES[w])
            models[w], sources[w] = m, src
        return models, charset, sources

    # ---- 1. ラベル(出荷時の重みで、確定値と整合する書き方のうち最尤のもの)
    label_models, charset, _ = load_init()
    label_engine = ParseqEngine.from_models(label_models, charset, device="cpu")
    prog("label", 0.02)
    user: dict[int, list[tuple[np.ndarray, str]]] = {256: [], 384: []}
    train_pages = plan["pages"]["train"]
    for n, pg in enumerate(train_pages):
        try:
            page, _py = load_page_input(pg["file"])
        except (OSError, ValueError, KeyError) as exc:
            raise WorkerError("pagedata_missing") from exc
        confirmed = {(c[0], c[1]): c for c in pg["cells"]}
        for cell in page.cells:
            c = confirmed.get((cell.row, cell.col))
            if c is None or cell.rec is None:
                continue
            _r, _c, column, kind, resolved = c
            if kind != "text" or not str(resolved).strip():  # 〃・矢印・波線・空欄は使わない
                continue
            for width, cols in WIDTH_COLUMNS.items():
                if column in cols:
                    label = choose_label(label_engine, cell.rec, column, str(resolved))
                    if label:
                        user[width].append((cell.rec, label))
        prog("label", 0.02 + 0.08 * (n + 1) / max(len(train_pages), 1))
    del label_engine, label_models
    _log(f"label: pages={len(train_pages)} cells256={len(user[256])} cells384={len(user[384])}")
    if not user[256] and not user[384]:
        raise WorkerError("no_training_cells")

    # ---- 2. 学習(出荷時の重みから。L2-SP の基準も出荷時の重み)
    models, charset, sources = load_init()
    rng = np.random.default_rng(seed)
    plans: dict[int, tuple[list, int, int]] = {}
    for width in (256, 384):
        if not user[width]:
            continue
        pool = rehearsal.get(width, []) if rehearsal else []
        samples, k = mix_rehearsal(user[width], pool, rng)
        plans[width] = (samples, epochs_for(len(samples)), k)
    total_epochs = sum(e for _, e, _ in plans.values())
    done = 0
    train_info: dict = {"samples": {}, "rehearsal": {}, "epochs": {}, "seconds": {}}

    def on_epoch(_msg: str) -> None:
        nonlocal done
        done += 1
        prog("train", 0.10 + 0.65 * done / max(total_epochs, 1))

    for width, (samples, epochs, k) in plans.items():
        cfg = FinetuneConfig(
            width=width,
            epochs=epochs,
            batch=BATCH,
            lr=LR,
            frozen_blocks=FROZEN_BLOCKS,
            l2sp=L2SP,
            seed=seed + width,
            device=device,
        )
        t0 = time.time()
        hist = finetune(models[width], charset, samples, cfg, log=on_epoch)
        train_info["samples"][str(width)] = len(samples)
        train_info["rehearsal"][str(width)] = k
        train_info["epochs"][str(width)] = epochs
        train_info["seconds"][str(width)] = round(time.time() - t0, 1)
        _log(
            f"train: width={width} samples={len(samples)} rehearsal={k} epochs={epochs} "
            f"loss_ar={hist['ar']:.3f} loss_refine={hist['refine']:.3f} seconds={time.time() - t0:.0f}"
        )

    # ---- 3. 評価(親と新しい世代を、評価用ページで認識し直す)
    ctx_plan = plan["context"]
    eval_pages = plan["pages"]["eval"]

    def context(prev_year: int | None) -> JobContext:
        return JobContext(
            prev_year=prev_year,
            accounts=list(ctx_plan["accounts"]),
            titles=list(ctx_plan["titles"]),
            dictionary=dict(ctx_plan["dictionary"]),
            amount_history={
                k: {int(a): int(n) for a, n in h.items()}
                for k, h in ctx_plan["amount_history"].items()
            },
        )

    inputs = []
    for pg in eval_pages:
        try:
            page, py = load_page_input(pg["file"])
        except (OSError, ValueError, KeyError) as exc:
            raise WorkerError("pagedata_missing") from exc
        conf = [ConfirmedCell(c[0], c[1], c[2], c[3], "", str(c[4])) for c in pg["cells"]]
        inputs.append((page, py, conf))

    def evaluate(
        rec: PageRecognizer,
    ) -> tuple[list[list[evalstats.CellObs]], list[list[bool]]]:
        """評価用ページごとの観測(確信度・正誤・要確認の理由)と、実際の要確認の判定。"""
        pages_obs: list[list[evalstats.CellObs]] = []
        pages_flags: list[list[bool]] = []
        for n, (page, py, conf) in enumerate(inputs):
            res = rec.recognize_page(page, context(py))
            obs, flags = [], []
            for r, t in zip(res.cells, conf, strict=True):
                idx = r.row * 4 + r.col
                obs.append(
                    evalstats.CellObs(
                        r.column,
                        r.kind,
                        float(r.confidence),
                        gate.same_value(t.column, r.value, t.resolved),
                        # 同上の参照元の要確認による理由は、しきい値で変わるので静的な理由に含めない(`evalstats.review_flags` が参照元から再現する)
                        bool(HARD_REASONS & (set(r.reasons) - {"ditto_source_uncertain"})),
                        page=n,
                        idx=idx,
                        nonblank=t.kind != "blank",
                        src=None if r.source is None else r.source[0] * 4 + r.source[1],
                    )
                )
                flags.append(bool(r.review))
            pages_obs.append(obs)
            pages_flags.append(flags)
        return pages_obs, pages_flags

    prog("eval", 0.76)
    parent_engine = ParseqEngine(models_dir=parent_dir, device=device)
    parent_cal = ConfidenceModel.load(parent_conf) if parent_conf.is_file() else None
    parent_obs, parent_flags = evaluate(PageRecognizer(parent_engine, clf, calibrator=parent_cal))
    del parent_engine
    parent_side = evalstats.side_from(parent_obs, parent_flags)
    _log(f"eval: parent {_fmt(parent_side)}")
    prog("eval", 0.84)
    cand_engine = ParseqEngine.from_models(models, charset, device=device)
    global_cal = ConfidenceModel.load(global_conf)
    cand_obs, cand_flags = evaluate(PageRecognizer(cand_engine, clf, calibrator=global_cal))
    del cand_engine
    # 要確認の判定をしきい値から再現できることの確認(再現が合わなければ、クロスフィッティングの数え方が信用できないので失敗にする)
    for pg, fl in zip(cand_obs, cand_flags, strict=True):
        if evalstats.review_flags(pg, global_cal) != fl:
            raise WorkerError("review_replay_mismatch")
    if parent_cal is not None:
        for pg, fl in zip(parent_obs, parent_flags, strict=True):
            if evalstats.review_flags(pg, parent_cal) != fl:
                raise WorkerError("review_replay_mismatch")
    side_global = evalstats.side_from(cand_obs, cand_flags)
    _log(f"eval: candidate(global thresholds) {_fmt(side_global)}")

    # ---- 4. 再較正(全体の較正から出発。観測した見逃しが目標を超えた列だけ厳しくする)+ クロスフィッティング
    prog("recalibrate", 0.92)
    n_confirmed = int(plan.get("n_confirmed_pages", len(train_pages) + len(eval_pages)))
    floor = (
        parent_cal if n_confirmed < recalibrate.FLOOR_PAGES else None
    )  # 確定ページが少ない間は親より緩めない
    cal, calibs, cf_flags = recalibrate.cross_fit(global_cal, cand_obs, floor=floor)
    summary = recalibrate.summarize(calibs)
    summary["floor"] = floor is not None
    cand_side = evalstats.side_from(
        cand_obs, cf_flags
    )  # 採否に使う成績: 各ページを除いて決めたしきい値での判定
    final_flags = [evalstats.review_flags(pg, cal) for pg in cand_obs]
    side_final = evalstats.side_from(
        cand_obs, final_flags
    )  # 参考: 全評価ページで決めたしきい値(同じページに合わせているので楽観的)
    n_cells = {c: len(eval_pages) * 29 for c in recalibrate.TARGETS}
    cal = recalibrate.warm_thresholds(cal, [o for pg in cand_obs for o in pg], n_cells)
    _log(f"eval: candidate(cross-fitted) {_fmt(cand_side)}")
    _log(f"eval: candidate(final thresholds, optimistic) {_fmt(side_final)}")
    if summary["changed"]:
        _log(f"recalibrate: changed columns={','.join(summary['changed'])}")

    # ---- 保存(テンソルと基本型だけの .pt。weights_only で読める)
    prog("save", 0.97)
    wdir = run_dir / WEIGHTS_SUBDIR
    wdir.mkdir(parents=True, exist_ok=True)
    for width, model in models.items():
        sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        save_converted(
            wdir / MODEL_FILES[width],
            sd,
            charset,
            width,
            {
                **sources[width],
                "finetuned": f"user data adaptation: run={run_id} "
                f"samples={train_info['samples'].get(str(width), 0)} "
                f"epochs={train_info['epochs'].get(str(width), 0)} l2sp={L2SP} lr={LR}",
            },
        )
    cal.save(wdir / "confidence.json")
    train_info.update(
        {"seconds_total": round(time.time() - t_start, 1), "device": device, "threads": threads}
    )
    prog("done", 1.0)
    return {
        "status": "ok",
        "weights_dir": str(wdir),
        "confidence": str(wdir / "confidence.json"),
        "eval": {
            "parent": parent_side.to_dict(),
            "candidate": cand_side.to_dict(),  # クロスフィッティングした判定での成績(ゲートが使う)
            "candidate_global": side_global.to_dict(),  # 参考: 全体の較正のしきい値のまま
            "candidate_final": side_final.to_dict(),  # 参考: 採用した場合に使うしきい値(楽観的)
        },
        "recalibration": summary,
        "train": train_info,
    }


def _fmt(side: gate.EvalSide) -> str:
    c = side.total
    k = side.calib
    extra = "" if k is None else f" nll={k.nll:.3f} expected_miss={k.expected_miss:.2f}"
    return (
        f"cells={c.n_cells} text={c.n_text} acc={c.accuracy:.3f} "
        f"review={c.review_rate:.3f} miss={c.n_miss}{extra}"
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: train_worker <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[0])
    with contextlib.suppress(AttributeError, OSError):
        os.nice(10)  # POSIX: 優先度を下げる(Windows は起動側が BELOW_NORMAL を指定する)
    try:
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        result = run(
            plan,
            run_dir,
            lambda s, f: _write_json(run_dir / "progress.json", {"stage": s, "progress": f}),
        )
    except WorkerError as exc:
        _log(f"failed: {exc.code}")
        _write_json(run_dir / "result.json", {"status": "failed", "reason": exc.code})
        return 1
    except Exception as exc:  # 型名と呼び出しの位置だけ記録する(例外の文面に値が入りうる)
        frames = [
            f"{Path(f.filename).name}:{f.lineno}" for f in traceback.extract_tb(exc.__traceback__)
        ]
        _log(f"failed: {type(exc).__name__} at {' > '.join(frames[-4:])}")
        _write_json(
            run_dir / "result.json", {"status": "failed", "reason": f"error_{type(exc).__name__}"}
        )
        return 1
    _write_json(run_dir / "result.json", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
