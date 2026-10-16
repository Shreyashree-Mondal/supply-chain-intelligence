"""Tests for PDF ingestion: text cleaning, sentence-aware chunking, and source/page metadata."""
import pytest

from src.llm import pdf_ingest
from src.llm.pdf_ingest import MAX_CHARS, _clean, _split, load_pdf_chunks


def test_clean_rejoins_hyphenated_words_and_collapses_whitespace():
    assert _clean("inven-\ntory   management\n\nis   key") == "inventory management is key"


def test_split_respects_max_chars_and_keeps_all_text():
    text = " ".join(f"Sentence number {i} about safety stock." for i in range(80))
    parts = _split(text)
    assert len(parts) > 1
    assert all(len(p) <= MAX_CHARS for p in parts)
    assert " ".join(parts) == text            # nothing lost or duplicated


def test_split_short_text_is_one_chunk():
    assert _split("Short note.") == ["Short note."]


def _make_pdf(path, pages):
    """Write a minimal text PDF (one string per page) so the test needs no real document."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(pages):
        objs.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>")
        stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{off:010d} 00000 n \n" for off in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def test_pdf_chunks_carry_source_and_page(tmp_path):
    _make_pdf(tmp_path / "sample.pdf", [
        "Page one explains what safety stock is and why planners hold it.",
        "Page two explains how the reorder point is calculated from lead time demand.",
    ])
    chunks = load_pdf_chunks(tmp_path)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "sample.pdf" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)          # ids are unique
    assert "reorder point" in next(c.text for c in chunks if c.page == 2)
