"""合成帳票の正解データ(raw / resolved)のスキーマと読み書き。

実データの文字起こしは作らない(開発中は実データの内容を扱わない)。正解データは合成器
(`yayoi_ocr.synth`)が出力する。スキーマは docs/ground_truth_schema.json。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class GTField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str
    resolved: str
    uncertain: bool = False
    note: str = ""


class GTRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: GTField
    account: GTField
    description: GTField
    amount: GTField


class GTPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    image: str
    title: GTField
    number: GTField
    rows: list[GTRow]


def load_ground_truth(path: str | Path) -> GTPage:
    return GTPage.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_ground_truth(gt: GTPage, path: str | Path) -> None:
    Path(path).write_text(gt.model_dump_json(indent=1), encoding="utf-8")
