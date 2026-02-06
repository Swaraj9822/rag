from text_utils import split_text


def test_split_text_prefers_structured_paragraph_boundaries():
    text = (
        "Section 1: Warranty\n"
        "Products include a 2-year warranty for manufacturer defects.\n\n"
        "Section 2: Support\n"
        "Support is available from Monday to Friday, 9am to 5pm.\n\n"
        "Section 3: Returns\n"
        "Returns are accepted within 30 days with proof of purchase."
    )

    chunks = split_text(text, chunk_size=120, overlap=0)

    assert len(chunks) == 3
    assert "Section 1: Warranty" in chunks[0]
    assert "Section 2: Support" in chunks[1]
    assert "Section 3: Returns" in chunks[2]


def test_split_text_splits_oversized_units_on_sentence_boundaries():
    text = " ".join([
        "This is sentence one.",
        "This is sentence two.",
        "This is sentence three.",
        "This is sentence four.",
    ])

    chunks = split_text(text, chunk_size=55, overlap=0)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 55 for chunk in chunks)


def test_split_text_falls_back_for_dense_text_without_sentences():
    dense = "x" * 130
    chunks = split_text(dense, chunk_size=50, overlap=0)

    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [50, 50, 30]
