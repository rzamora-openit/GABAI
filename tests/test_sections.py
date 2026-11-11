from RAG.sections import Section, parse_markdown, pop_last_section, split_section

MD = "# Doc\n\n## Admission\n\nApply early.\n\n## Fees\n\nPay on time."


def test_parse_markdown_sections():
    sections = parse_markdown(MD, "Doc")
    assert [(s.title, s.body, s.order) for s in sections] == [
        ("Admission", "Apply early.", 0),
        ("Fees", "Pay on time.", 1),
    ]


def test_no_headings_becomes_one_section():
    assert [(s.title, s.body) for s in parse_markdown("just text", "Doc")] == [("Doc", "just text")]


def test_text_before_first_heading_gets_default_title():
    assert parse_markdown("intro\n\n## A\n\nbody", "Doc")[0].title == "Doc"


def test_empty_sections_dropped_and_reordered():
    assert [s.order for s in parse_markdown("## A\n\n## B\n\nb", "Doc")] == [0]


def test_pop_last_section():
    kept, last = pop_last_section(MD)
    assert last == "## Fees\n\nPay on time."
    assert "Fees" not in kept and "Admission" in kept


def test_pop_last_section_without_heading():
    assert pop_last_section("plain") == ("plain", "")


def test_split_section_respects_limit_and_keeps_all_words():
    body = "\n\n".join(["word " * 40] * 5)
    pieces = split_section(Section("T", body, 0), 500)
    assert all(len(p) <= 500 for p in pieces)
    assert " ".join(pieces).split() == body.split()


def test_split_section_cuts_oversized_paragraph_at_whitespace():
    pieces = split_section(Section("T", "ab " * 400, 0), 100)
    assert all(len(p) <= 100 for p in pieces)
    assert len(pieces) > 1
