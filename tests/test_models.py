from __future__ import annotations

from yayoi_ocr.models import (
    BBox,
    Cell,
    CellFlag,
    CellStatus,
    ColumnKind,
    Document,
    ImageRef,
    Page,
    Row,
)


def test_document_roundtrip():
    cell = Cell(
        column=ColumnKind.DESCRIPTION,
        bbox=BBox(x=1, y=2, w=3, h=4),
        raw_text="〃",
        resolved_value="サンプル商事",
        char_confidences=[0.9],
        flags={CellFlag.DITTO},
    )
    page = Page(
        index=0,
        image=ImageRef(width=10, height=20),
        rows=[Row(index=0, cells=[cell])],
    )
    doc = Document(id="d1", pages=[page])
    again = Document.model_validate_json(doc.model_dump_json())
    c = again.pages[0].rows[0].cell(ColumnKind.DESCRIPTION)
    assert c.raw_text == "〃" and c.resolved_value == "サンプル商事"
    assert CellFlag.DITTO in c.flags
    assert c.status == CellStatus.UNCONFIRMED
