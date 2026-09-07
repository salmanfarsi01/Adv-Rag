from ocr_pipeline.chunking import chunk_document
from ocr_pipeline.structured import LayoutBlock, StructuredDocument, StructuredPage


def test_chunking_preserves_table_blocks_and_source_metadata():
    document = StructuredDocument("invoice_00234", [StructuredPage(1, [
        LayoutBlock("Invoice total is 42 dollars.", (10, 10, 300, 30), 95, "paragraph"),
        LayoutBlock("Item  Qty  Total", (10, 60, 300, 80), 91, "table_cell"),
        LayoutBlock("Widget  2  42", (10, 85, 300, 105), 88, "table_cell"),
    ])])

    chunks = chunk_document(document)

    assert len(chunks) == 3
    assert chunks[0].chunk_id == "invoice_00234:p1:c0"
    assert chunks[1].block_types == ("table_cell",)
    assert chunks[1].page_num == 1
    assert chunks[1].bbox == (10, 60, 300, 80)
    assert chunks[1].ocr_confidence == 91
