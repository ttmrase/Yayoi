"""GitHub Actions のワークフローの安全の約束(ファイルの文面の検査。構造の検査だけ PyYAML があれば行う): 外部アクションはコミットの SHA で固定・
trial-build は読み取りだけ・trial-publish は手動(workflow_dispatch)だけで、書き込みの権限は publish ジョブにだけ・PAT や secrets を使わない。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"
FILES = sorted(WF.glob("*.yml"))
USES = re.compile(r"^\s*-?\s*uses:\s*(\S+)", re.M)


def text(name: str) -> str:
    """コメントを除いた文面(説明のコメントに権限の語が出ても数えない)。"""
    t = (WF / name).read_text(encoding="utf-8")
    return re.sub(r"(?m)(^|\s)#.*$", "", t)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_external_action_is_pinned_to_a_commit_sha(path):
    uses = USES.findall(path.read_text(encoding="utf-8"))
    assert uses
    for u in uses:
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", u), (
            f"{path.name}: {u} がコミットの SHA で固定されていない"
        )


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_workflow_uses_secrets_or_privileged_triggers(path):
    t = path.read_text(encoding="utf-8")
    assert "pull_request_target" not in t and "workflow_run:" not in t
    assert not re.findall(r"secrets\.(?!GITHUB_TOKEN)\w+", t)


def test_trial_build_only_reads_and_never_publishes():
    t = text("trial-build.yml")
    assert re.search(r"^permissions:\n  contents: read\n  actions: read", t, re.M)  # 読み取りだけ
    assert not re.search(r":\s*write\b", t)
    assert "gh release" not in t and "git push" not in t and "git tag" not in t
    # 起動は、要求ファイルの変更の push と、手動だけ
    on = t.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert (
        re.search(r"paths: \[\"\.github/trial-build-request\"\]", on) and "workflow_dispatch" in on
    )
    assert "schedule" not in on and "pull_request" not in on


def test_trial_build_checkouts_do_not_persist_credentials():
    for name in ("trial-build.yml", "trial-publish.yml"):
        t = text(name)
        assert t.count("actions/checkout@") == t.count("persist-credentials: false"), name


def test_trial_publish_runs_only_when_a_person_clicks_run_workflow():
    t = text("trial-publish.yml")
    on = t.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert on.strip() == "workflow_dispatch:"  # 入力もなし。押すこと自体が承認
    assert re.search(r"^permissions: \{\}\n", t, re.M)  # 既定の権限はなし。ジョブごとに付ける


def test_write_permission_is_given_only_to_the_publish_job():
    t = text("trial-publish.yml")
    writes = [m.start() for m in re.finditer(r"contents: write", t)]
    assert len(writes) == 1
    jobs = t.split("\njobs:\n", 1)[1]
    publish = jobs.split("\n  publish:\n", 1)
    assert (
        len(publish) == 2
        and "contents: write" in publish[1]
        and "contents: write" not in publish[0]
    )


def test_trial_publish_never_overwrites_and_rolls_back_a_failed_publication():
    t = text("trial-publish.yml")
    assert (
        t.count("must not exist") + t.count("must still not exist") == 2
    )  # verify と publish の両方で、既存のタグ・Release を拒否
    assert "--clobber" not in t and "--force" not in t
    assert "gh release delete" in t and "--cleanup-tag" in t
    assert "--prerelease" in t and "--target" in t


def test_trial_eval_only_reads_and_does_not_keep_the_evaluation_pages():
    t = text("trial-eval.yml")
    assert re.search(r"^permissions:\n  contents: read\n  actions: read\n", t, re.M)
    assert not re.search(r":\s*write\b", t) and "gh release" not in t and "git push" not in t
    # 評価ページは artifact にしない(結果の JSON だけを保存する)
    uploads = re.findall(r"upload-artifact@[0-9a-f]{40}[^\n]*\n\s+with:\n\s+name: ([^\n]+)\n", t)
    assert len(uploads) == 1 and "trial-eval-result" in uploads[0]


def test_trial_publish_requires_the_quality_evaluation_and_the_expected_runs():
    t = text("trial-publish.yml")
    assert t.count("--eval-dir") == 2 and t.count("name: trial-eval-result") == 2
    # ワークフローのパス・ブランチ・コミットを確かめる(W-2)
    assert ".github/workflows/trial-build.yml" in t and ".github/workflows/trial-eval.yml" in t
    assert t.count("claude/handwritten-document-ocr-sgf76h") >= 2


def _yaml():
    # PyYAML は ml の extra にだけある(pyproject.toml は、重みの再利用の照合の対象なので、ここでは変えない)。無い環境では、構造の検査を飛ばす
    return pytest.importorskip("yaml")


def _build_jobs() -> dict:
    return _yaml().safe_load(text("trial-build.yml"))["jobs"]


def test_a_manual_build_defaults_to_the_small_test_run_not_the_distribution_build():
    on = _yaml().safe_load(text("trial-build.yml"))[True]
    assert on["workflow_dispatch"]["inputs"]["mode"]["default"] == "mini"


def test_the_reuse_path_skips_the_heavy_stages_and_runs_only_the_tail():
    jobs = _build_jobs()
    for name in ("shards", "base-weights"):
        assert jobs[name]["if"] == "needs.plan.outputs.reuse == ''"  # 再利用のときは作らない
    # 前の段階が skipped でも進む(暗黙の success() だと、後続の job まで飛ばされる。通し試験で確認)が、失敗・取り消しでは進まない
    for name, needed in (
        ("confidence", ["plan"]),
        ("bundle", ["confidence"]),
        ("smoke-windows", ["bundle"]),
        ("finalize", ["bundle", "smoke-windows"]),
    ):
        cond = jobs[name]["if"]
        assert "!cancelled()" in cond and "!failure()" in cond, name
        assert all(f"needs.{n}.result == 'success'" in cond for n in needed), name


def test_the_reuse_path_checks_the_earlier_run_before_using_anything_from_it():
    plan_steps = _build_jobs()["plan"]["steps"]
    runs = "\n".join(str(s.get("run", "")) for s in plan_steps)
    assert "run-check" in runs and ".github/workflows/trial-build.yml" in runs
    assert "reuse-check" in runs  # コマンドの一致とコードの変更の照合
    ck = next(i for i, s in enumerate(plan_steps) if "reuse-check" in str(s.get("run", "")))
    assert plan_steps[0]["with"]["fetch-depth"] == 0  # コミットの差を調べるため、履歴が要る
    assert ck > next(i for i, s in enumerate(plan_steps) if s.get("id") == "reuse")


def test_the_only_artifacts_taken_from_the_earlier_run_carry_its_run_id_and_the_new_confidence_comes_from_this_run():
    steps = _build_jobs()["bundle"]["steps"]
    downloads = [
        s for s in steps if str(s.get("uses", "")).startswith("actions/download-artifact@")
    ]
    from_earlier = [s for s in downloads if "run-id" in s["with"]]
    assert {s["with"].get("name") or s["with"].get("pattern") for s in from_earlier} == {
        "w.base",
        "w.celltype",
        "w.adapted",
        "meta.*",
    }
    assert all(s["with"]["run-id"] == "${{ needs.plan.outputs.reuse }}" for s in from_earlier)
    assert all("github-token" in s["with"] for s in from_earlier)
    # 確信度(w.confidence)とその記録(meta.fit)は、必ずこの run のもの(前の run の古い較正を使わない)
    here = [s for s in downloads if "run-id" not in s["with"] and s.get("if", "").endswith("!= ''")]
    assert {s["with"].get("name") for s in here} == {"w.confidence", "meta.fit"}
