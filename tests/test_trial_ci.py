"""配布物の GitHub Actions ビルド(scripts/trial_ci.py): 計画(合成データのシャード)・出所の文書の生成・公開の内容(JSON)と成果物の照合。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("trial_ci", ROOT / "scripts" / "trial_ci.py")
tci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tci)
vspec = importlib.util.spec_from_file_location(
    "verify_trial_bundle", ROOT / "scripts" / "verify_trial_bundle.py"
)
vtb = importlib.util.module_from_spec(vspec)
vspec.loader.exec_module(vtb)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---- 計画


def test_the_full_plan_shards_cover_exactly_the_pages_of_the_documented_recipes():
    p = tci.plan("full")
    assert len(p["shards"]) == 16 and all(s["n"] <= 100 for s in p["shards"])
    pages: dict[str, set[int]] = {}
    writers: dict[str, set[int]] = {}
    for s in p["shards"]:
        seeds = set(range(s["seed0"], s["seed0"] + s["n"]))
        assert not (pages.setdefault(s["dataset"], set()) & seeds)  # シャード同士で重ならない
        pages[s["dataset"]] |= seeds
        writers.setdefault(s["dataset"], set()).update(range(s["writer0"], s["writer0"] + s["n"]))
        assert s["extra"][:3] == ["--font-exclude", "yusei", "klee"]
    expect = {
        "train": [(1000000, 300), (1100000, 100), (1200000, 100)],
        "calib": [(2000000, 100)],
        "calib500": [(7000000, 500)],
        "train_gen": [(8000000, 300)],
        "calib_gen": [(9000000, 200)],
    }
    for name, runs in expect.items():
        assert pages[name] == {s0 + i for s0, n in runs for i in range(n)}, name
    assert sum(len(v) for v in pages.values()) == 1600
    # 学習・較正・評価の分離: データセット間で seed・書き手 ID が重ならない
    names = sorted(pages)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (pages[a] & pages[b]) and not (writers[a] & writers[b])


def test_the_full_plan_uses_the_documented_seeds_and_thread_counts():
    p = tci.plan("full")
    assert p["celltype"] == ["--epochs", "8", "--threads", "4"]
    assert p["adapt"] == ["--l2sp", "0.1", "--epochs", "3", "--threads", "3"]
    assert p["dump"] == [
        "--threads",
        "3",
    ]  # ダンプは --threads 3 のときだけ、元のダンプとビット一致した
    g = next(s for s in p["shards"] if s["dataset"] == "train_gen")
    assert (
        "--vocab" in g["extra"]
        and "general-train" in g["extra"]
        and "--strength-range" in g["extra"]
    )


def test_the_mini_plan_is_small_and_never_shares_a_name_with_the_full_artifact():
    p = tci.plan("mini")
    assert len(p["shards"]) == 7 and sum(s["n"] for s in p["shards"]) == 84
    assert tci.plan("full")["dumps"] == p["dumps"]
    ids = [s["id"] for s in p["shards"]]
    assert len(set(ids)) == len(ids) and all(i.count(".") == 2 for i in ids)


def test_stage_commands_reproduce_the_documented_commands(tmp_path):
    d, m, e = tmp_path / "d", tmp_path / "m", tmp_path / "e"
    (cmd,) = tci.stage_commands("celltype", "full", d, m, e)
    assert cmd[2:4] == ["scripts/train_celltype.py", "--train"] and "--epochs" in cmd
    cmds = tci.stage_commands("fit", "full", d, m, e)
    assert len(cmds) == 2 and all("--margins" in c and "amount=0.4" in c for c in cmds)
    (dump,) = tci.stage_commands("dump", "full", d, m, e, dataset="calib500", weights="adapted")
    assert "--models-dir" in dump and str(m / "adapted") in dump
    (base,) = tci.stage_commands("dump", "full", d, m, e, dataset="calib500", weights="base")
    assert "--models-dir" not in base


# ---- 出所の文書


def fake_models(root: Path) -> dict[str, str]:
    out = {}
    for rel in tci.MODEL_FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"weights {rel}".encode())
        out[f"models/{rel}"] = sha(p.read_bytes())
    return out


def fake_summaries(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, (name, seeds) in enumerate(
        [("train", [1, 2, 3]), ("train_gen", [10, 11]), ("calib", [20])]
    ):
        (d / f"summary.{name}.{i}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "pages": len(seeds),
                    "bad": [],
                    "cells": 116 * len(seeds),
                    "seed_set": seeds,
                    "writer_set": [100 + i * 10 + k for k in range(len(seeds))],
                    "fonts": {"Yomogi-Regular": len(seeds)},
                }
            ),
            encoding="utf-8",
        )


def test_the_provenance_document_carries_the_hashes_of_the_weights_actually_built(tmp_path):
    expected = fake_models(tmp_path / "models")
    fake_summaries(tmp_path / "sum")
    infos = [
        {
            "stage": "celltype",
            "id": "celltype",
            "seconds": 12.0,
            "commands": ["python scripts/train_celltype.py"],
            "cpu": "TestCPU",
            "versions": {"torch": "2.15.0"},
        }
    ]
    template = (ROOT / "docs" / "model_provenance.md").read_text(encoding="utf-8")
    md = tci.build_provenance(
        template,
        tmp_path / "models",
        tmp_path / "sum",
        infos,
        run_id="123",
        commit="a" * 40,
        mode="full",
        ref="b" * 40,
        now="2026-01-01 00:00 UTC",
    )
    table = vtb.provenance_table(md)
    assert table == expected  # B4 が照合する表 = 実際の重みの SHA256(9 ファイル)
    assert "run 123" in md and "a" * 40 in md and "TestCPU" in md
    assert (
        "この束の重みには行っていません" in md
    )  # 拒否リストの走査(C4・C6)をしていないことを明記する
    assert (
        "### 実行したスクリプトとコミット" in md and "`f023270`" not in md
    )  # コンテナの実行記録は置き換えた


def test_the_provenance_generation_refuses_overlapping_datasets(tmp_path):
    fake_models(tmp_path / "models")
    fake_summaries(tmp_path / "sum")
    f = next((tmp_path / "sum").glob("summary.calib.*.json"))
    s = json.loads(f.read_text())
    s["seed_set"] = [1]  # train と重なる
    f.write_text(json.dumps(s))
    template = (ROOT / "docs" / "model_provenance.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="重なり"):
        tci.build_provenance(
            template,
            tmp_path / "models",
            tmp_path / "sum",
            [],
            run_id="1",
            commit="c" * 40,
            mode="full",
            ref="d" * 40,
            now="x",
        )


# ---- 公開の内容と成果物の照合


def make_release(tmp_path: Path, **override) -> tuple[Path, Path, Path]:
    art = tmp_path / "artifact"
    art.mkdir()
    zip_bytes = b"zip bytes"
    digest = sha(zip_bytes)
    (art / "yayoi-ocr-trial-0.1.0.zip").write_bytes(zip_bytes)
    (art / "yayoi-ocr-trial-0.1.0.zip.sha256").write_text(f"{digest}  yayoi-ocr-trial-0.1.0.zip\n")
    (art / "model_provenance.generated.md").write_text("x")
    info = {
        "mode": "full",
        "run_id": "99",
        "commit": "e" * 40,
        "zip": {"sha256": digest},
        "verify_bundle": {"ok": True},
        "smoke_windows": {"ok": True},
        "github": {"run_id": "99", "sha": "e" * 40},
    }
    info.update(override.pop("info", {}))
    eval_override = override.pop("eval", {})
    (art / "build-info.json").write_text(json.dumps(info))
    spec = {
        "tag": "trial-v0.1.0",
        "commit": "e" * 40,
        "run_id": "99",
        "eval_run_id": "100",
        "zip_name": "yayoi-ocr-trial-0.1.0.zip",
        "zip_sha256": digest,
        "title": "試用版",
    }
    spec.update(override)
    ev = {
        "zip_sha256": digest,
        "build_run_id": "99",
        "build_commit": "e" * 40,
        "eval_run_id": "100",
        "mode": "full",
        "passed": True,
        "criteria_version": tci.EVAL_CRITERIA_VERSION,
        "checks": [{"id": i, "ok": True} for i in tci.required_eval_check_ids()],
    }
    ev.update(eval_override)
    (tmp_path / "eval").mkdir(exist_ok=True)
    (tmp_path / "eval" / "eval-result.json").write_text(json.dumps(ev))
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    notes = tmp_path / "notes.md"
    notes.write_text(f"SHA256: `{digest}`\n")
    return sp, notes, art


def check(sp, notes, art) -> int:
    class A:
        json = str(sp)
        dir = str(art)
        eval_dir = str(sp.parent / "eval")

    A.notes = str(notes)
    return tci.cmd_release_check(A)


def test_a_consistent_release_spec_and_artifact_pass(tmp_path):
    sp, notes, art = make_release(tmp_path)
    assert check(sp, notes, art) == 0


@pytest.mark.parametrize(
    "override",
    [
        {"info": {"mode": "mini"}},  # mini の束は公開しない
        {"info": {"run_id": "98", "github": {"run_id": "98", "sha": "e" * 40}}},
        {"info": {"smoke_windows": {"ok": False}}},
        {"info": {"verify_bundle": {"ok": False}}},
        {"info": {"github": {"run_id": "99", "sha": "f" * 40}}},  # ビルドのコミットが JSON と違う
    ],
)
def test_an_artifact_that_does_not_match_or_failed_its_checks_is_refused(tmp_path, override):
    sp, notes, art = make_release(tmp_path, **override)
    assert check(sp, notes, art) == 1


def test_a_tampered_zip_is_refused(tmp_path):
    sp, notes, art = make_release(tmp_path)
    (art / "yayoi-ocr-trial-0.1.0.zip").write_bytes(b"different")
    assert check(sp, notes, art) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"tag": "v0.1.0"},
        {"commit": "abc123"},  # 短いコミット ID は不可
        {"run_id": "abc"},
        {"zip_name": "yayoi-ocr-trial-0.1.0-MINI.zip"},
        {"zip_sha256": "A" * 64},
    ],
)
def test_an_invalid_release_spec_is_refused(tmp_path, override):
    sp, notes, _art = make_release(tmp_path, **override)
    with pytest.raises(SystemExit):
        tci.load_release_spec(sp, notes)


def test_the_release_notes_must_carry_exactly_the_sha256_of_the_spec(tmp_path):
    sp, notes, _art = make_release(tmp_path)
    notes.write_text("SHA256: `" + "0" * 64 + "`\n")
    with pytest.raises(SystemExit, match="SHA256"):
        tci.load_release_spec(sp, notes)


# ---- ビルドの記録(augment)


def run_augment(tmp_path, smoke: dict | None, verify_text: str):
    import argparse

    z = tmp_path / "yayoi-ocr-trial-0.1.0.zip"
    z.write_bytes(b"zip bytes")
    info = tmp_path / "info.json"
    info.write_text(json.dumps({"mode": "full", "run_id": "1", "commit": "a" * 40}))
    ver = tmp_path / "verify.txt"
    ver.write_text(verify_text, encoding="utf-8")
    ns = argparse.Namespace(
        info=str(info),
        out=str(tmp_path / "out.json"),
        zip=str(z),
        verify_output=str(ver),
        smoke=None,
    )
    if smoke is not None:
        sm = tmp_path / "smoke.json"
        sm.write_text(json.dumps(smoke))
        ns.smoke = str(sm)
    assert tci.cmd_augment(ns) == 0
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8")), sha(b"zip bytes")


def test_the_windows_smoke_result_counts_only_for_the_exact_zip_that_was_tested(tmp_path):
    out, digest = run_augment(
        tmp_path, {"passed": True, "zip_sha256": sha(b"zip bytes")}, "[OK] B1\n結果: OK\n"
    )
    assert out["smoke_windows"]["ok"] is True and out["verify_bundle"]["ok"] is True
    assert out["zip"]["sha256"] == digest and out["zip"]["name"] == "yayoi-ocr-trial-0.1.0.zip"


def test_a_smoke_result_for_a_different_zip_or_a_failed_check_is_not_ok(tmp_path):
    out, _ = run_augment(tmp_path, {"passed": True, "zip_sha256": "0" * 64}, "結果: NG\n")
    assert out["smoke_windows"]["ok"] is False and out["verify_bundle"]["ok"] is False
    out2, _ = run_augment(
        tmp_path, {"passed": False, "zip_sha256": sha(b"zip bytes")}, "結果: OK\n"
    )
    assert out2["smoke_windows"]["ok"] is False


# ---- 評価(trial-eval)と公開の条件


def stats(miss_rate: float, acc: float, n: int = 5800) -> dict:
    return {
        "n": n,
        "n_nonblank": int(n * 0.66),
        "miss": round(miss_rate * n),
        "miss_rate": miss_rate,
        "accuracy_nonblank": acc,
        "review_rate_nonblank": 0.8,
    }


def good_result(weights: str, pages: int = 200) -> dict:
    base = tci.BASELINE_ACCURACY[weights]
    columns = {c: stats(0.0005, base[c] - 0.01) for c in base}
    return {
        "weights": weights,
        "pages": pages,
        "overall": stats(0.002, 0.6, 23200),
        "columns": columns,
        "per_dataset": {
            "final_std": {"pages": 50, "overall": stats(0.002, 0.6), "columns": columns}
        },
    }


def test_the_eval_plan_is_the_documented_final_holdout_and_disjoint_from_training():
    shards = tci.plan("full")["eval_shards"]
    assert [s["dataset"] for s in shards] == [
        "final_std",
        "final_gen",
        "final_uf_std",
        "final_uf_gen",
    ]
    assert sum(s["n"] for s in shards) == 200
    train_pages = {
        s0 + i for s in tci.plan("full")["shards"] for s0 in [s["seed0"]] for i in range(s["n"])
    }
    train_writers = {
        w0 + i for s in tci.plan("full")["shards"] for w0 in [s["writer0"]] for i in range(s["n"])
    }
    for s in shards:
        assert not (set(range(s["seed0"], s["seed0"] + s["n"])) & train_pages)
        assert not (set(range(s["writer0"], s["writer0"] + s["n"])) & train_writers)
    uf = [s for s in shards if "uf" in s["dataset"]]
    assert all("--font-include" in s["extra"] for s in uf)  # 未知のフォントの組を含む
    assert tci.plan("mini")["eval_shards"][0]["n"] == 6


def test_a_result_that_meets_the_criteria_passes_for_both_weights():
    out = tci.merge_eval(good_result("base"), good_result("adapted"), {"mode": "full"})
    assert out["passed"] is True and all(c["ok"] for c in out["checks"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["overall"].update(miss_rate=0.0051),  # 全体の見逃し率が 0.5% を超える
        lambda r: r["columns"]["amount"].update(miss_rate=0.0011),  # 金額が 0.1% を超える
        lambda r: r["columns"]["date"].update(miss_rate=0.0052),  # 列ごと: 日付が 0.5% を超える
        lambda r: r["columns"]["account"].update(miss_rate=0.0052),  # 科目が 0.5% を超える
        lambda r: r["columns"]["description"].update(miss_rate=0.0052),  # 摘要が 0.5% を超える
        lambda r: r["columns"]["description"].update(
            accuracy_nonblank=tci.BASELINE_ACCURACY["adapted"]["description"] - 0.031
        ),  # コンテナの値から 3 ポイントを超えて劣化
    ],
)
def test_a_result_that_misses_a_criterion_is_rejected(mutate):
    adapted = good_result("adapted")
    mutate(adapted)
    out = tci.merge_eval(good_result("base"), adapted, {"mode": "full"})
    assert out["passed"] is False and any(not c["ok"] for c in out["checks"])


def test_fewer_than_200_pages_cannot_pass_a_full_evaluation():
    out = tci.merge_eval(good_result("base", 50), good_result("adapted", 50), {"mode": "full"})
    assert out["passed"] is False


def test_the_per_column_miss_limits_apply_to_both_weights_and_to_the_limit_exactly():
    # ちょうど上限は合格(<=)。列ごとの上限は、全体の上限とは別に判定される(全体が良くても、1 つの列が超えれば不合格)
    ok = good_result("adapted")
    ok["columns"]["account"].update(miss_rate=0.005)
    ok["columns"]["amount"].update(miss_rate=0.001)
    assert tci.merge_eval(good_result("base"), ok, {"mode": "full"})["passed"] is True
    for weights in ("base", "adapted"):
        bad = good_result(weights)
        bad["columns"]["account"].update(miss_rate=0.0051)  # 全体は 0.2% のまま
        pair = (bad, good_result("adapted")) if weights == "base" else (good_result("base"), bad)
        out = tci.merge_eval(*pair, {"mode": "full"})
        assert out["passed"] is False
        failed = [c["id"] for c in out["checks"] if not c["ok"]]
        assert failed == [f"{weights}:miss:account"]


def test_the_checks_cover_every_required_id_and_the_result_carries_the_criteria_version():
    out = tci.merge_eval(good_result("base"), good_result("adapted"), {"mode": "full"})
    assert sorted(c["id"] for c in out["checks"]) == sorted(tci.required_eval_check_ids())
    assert out["criteria_version"] == tci.EVAL_CRITERIA_VERSION
    required = tci.required_eval_check_ids()
    for w in ("base", "adapted"):
        for col in ("date", "account", "description", "amount"):
            assert f"{w}:miss:{col}" in required and f"{w}:acc:{col}" in required


def test_the_eval_table_prints_counts_and_rates_per_column_but_no_contents():
    lines = tci.format_eval_table(good_result("adapted"))
    text = "\n".join(lines)
    for label in ("日付", "科目", "摘要", "金額", "全体", "final_std"):
        assert label in text
    assert "見逃し 3/5800(0.050%)" in text  # 件数 / セル数(割合)


def test_the_release_rejects_an_evaluation_without_the_per_column_criteria(tmp_path):
    sp, notes, art = make_release(tmp_path)
    assert check(sp, notes, art) == 0
    old_ids = [i for i in tci.required_eval_check_ids() if ":miss:" in i and ":overall" not in i]
    cases = {
        "version_1": {"criteria_version": 1},
        "no_version": {"criteria_version": None},
        "no_column_checks": {
            "checks": [
                {"id": i, "ok": True}
                for i in tci.required_eval_check_ids()
                if i not in ("adapted:miss:account", "adapted:miss:date")
            ]
        },
        "a_check_failed": {
            "checks": [
                {"id": i, "ok": i != "base:miss:description"} for i in tci.required_eval_check_ids()
            ]
        },
        "old_format": {"checks": [{"name": "x", "ok": True}]},
    }
    assert old_ids  # 列ごとの判定が必要な ID に含まれている
    for k, ev in cases.items():
        d = tmp_path / f"case_{k}"
        d.mkdir()
        sp2, notes2, art2 = make_release(d, eval=ev)
        assert check(sp2, notes2, art2) == 1, k


def test_the_release_needs_a_passing_evaluation_of_exactly_this_zip(tmp_path):
    sp, notes, art = make_release(tmp_path)
    assert check(sp, notes, art) == 0
    for k, v in (
        ("passed", False),
        ("zip_sha256", "0" * 64),
        ("build_run_id", "98"),
        ("build_commit", "f" * 40),
        ("eval_run_id", "101"),
        ("mode", "mini"),
    ):
        d = tmp_path / f"case_{k}"
        d.mkdir()
        sp2, notes2, art2 = make_release(d, eval={k: v})
        assert check(sp2, notes2, art2) == 1, k


def test_a_release_spec_without_an_eval_run_is_invalid(tmp_path):
    sp, notes, _art = make_release(tmp_path, eval_run_id="abc")
    with pytest.raises(SystemExit):
        tci.load_release_spec(sp, notes)


def test_run_check_accepts_only_the_expected_workflow_branch_and_commit(tmp_path):
    import argparse

    run = {
        "id": 1,
        "path": ".github/workflows/trial-build.yml",
        "head_branch": "claude/handwritten-document-ocr-sgf76h",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "head_sha": "a" * 40,
    }

    def go(**override):
        r = {**run, **override}
        f = tmp_path / "run.json"
        f.write_text(json.dumps(r))
        return tci.cmd_run_check(
            argparse.Namespace(
                run=str(f),
                workflow_path=".github/workflows/trial-build.yml",
                branch="claude/handwritten-document-ocr-sgf76h",
                sha="a" * 40,
            )
        )

    assert go() == 0
    assert go(path=".github/workflows/ci.yml") == 1
    assert go(head_branch="main") == 1
    assert go(event="pull_request") == 1
    assert go(conclusion="failure") == 1
    assert go(head_sha="b" * 40) == 1
