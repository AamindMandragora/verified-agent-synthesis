import itertools

from synthesis.evaluate.benchmarks.common.delimiter_hygiene import (
    scrub_free_text,
    Span, banned_ids_outside_span, build_delimiter_token_sets, free_text_is_clean, is_opener_text, walk_spans,
)

VOCAB = {0: "<<", 1: " <<", 2: ">>", 3: " >>", 4: "<<<", 5: ")<<", 6: "<", 7: ">", 8: "<=", 9: ">=",
         10: "x", 11: " 3", 12: "<<(", 13: ">>\n", 14: "\n<<", 15: "<< "}


def test_walk_finds_closed_and_open_spans():
    w = walk_spans("a <<1+2>> b <<3")
    assert w.spans == (Span("1+2", True), Span("3", False))
    assert w.ends_inside


def test_a_runtime_opened_span_is_read_back_out_of_the_text():
    # The runtime opens the span by putting a literal "<<" in the output, so
    # there is nothing special to read: it is one span like any other.
    assert walk_spans("<<SELECT 1").spans == (Span("SELECT 1", False),)
    assert walk_spans("<<CCO>>").spans == (Span("CCO", True),)


def test_walk_matches_evaluator_regex_on_balanced_text():
    import re
    text = "x <<a>> y <<b c>> z"
    assert [s.content for s in walk_spans(text).spans] == re.findall(r"<<(.*?)>>", text, flags=re.DOTALL)


def test_open_inside_a_span_is_content_not_a_new_span():
    assert walk_spans("<<a<<b>>").spans == (Span("a<<b", True),)


def test_stray_close_is_flagged():
    assert not free_text_is_clean("a >> b")
    assert free_text_is_clean("a <<b>> c")
    assert not free_text_is_clean("<<a>> >>")


def test_token_classes():
    s = build_delimiter_token_sets(VOCAB)
    assert s.opener_ids == {0, 1, 14}
    assert s.exact_opener_ids == {0}
    assert s.always_banned_ids == {2, 3, 4, 5, 12, 13, 15}
    assert {6, 8, 0, 4, 12, 15} <= s.starts_with_lt_ids
    assert is_opener_text(" <<") and not is_opener_text("<< ") and not is_opener_text("<<<")


def test_tail_aware_ban_blocks_cross_token_delimiters():
    s = build_delimiter_token_sets(VOCAB)
    assert 6 not in banned_ids_outside_span(s, "a ")
    assert {6, 8, 0} <= banned_ids_outside_span(s, "a <")
    assert {1, 14} <= banned_ids_outside_span(s, "a ")  # only the exact "<<" token may open a span
    assert 0 not in banned_ids_outside_span(s, "a ")
    assert {7, 9} <= banned_ids_outside_span(s, "a >")


def test_no_allowed_sequence_can_create_a_stray_or_malformed_delimiter():
    """Exhaustive over short sequences: sampling only allowed tokens outside spans keeps free text clean and
    every span opener exactly `<<` (never `<<<`)."""
    s = build_delimiter_token_sets(VOCAB)
    for seq in itertools.product(VOCAB, repeat=3):
        text, ok = "", True
        for tid in seq:
            w = walk_spans(text)
            if w.ends_inside:
                break  # inside a span the parser decides, not this rule
            if tid in banned_ids_outside_span(s, w.free_text_tail):
                ok = False
                break
            text += VOCAB[tid]
        if not ok:
            continue
        assert free_text_is_clean(text), (seq, text)
        assert "<<<" not in text, (seq, text)


def test_scrub_drops_the_second_char_of_a_stray_close():
    assert scrub_free_text("", "a >> b") == "a > b"
    assert scrub_free_text("", "a >>> b") == "a > b"
    assert scrub_free_text("x >", "> y") == " y"


def test_scrub_drops_an_open_that_would_straddle_the_boundary():
    assert scrub_free_text("x <", "<3") == "3"
    assert scrub_free_text("x <", " <<1+1") == " <<1+1"


def test_scrub_leaves_clean_text_and_openers_alone():
    for t in ["plain", "a < b > c", "so <<", "-> x"]:
        assert scrub_free_text("", t) == t


def test_scrub_is_idempotent_and_result_is_clean():
    for tail, t in itertools.product(["", "<", ">", "a"], ["", ">", ">>", ">>>>", "<", "><>>", ">a>>"]):
        once = scrub_free_text(tail, t)
        assert scrub_free_text(tail, once) == once
        assert ">>" not in (tail[-1:] + once)
