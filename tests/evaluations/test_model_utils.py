from evaluation.common.model_utils import (
    _dedupe_token_ids_by_decoded_string,
    _valid_tokens_ids_logits_py,
)


class _FakeTokenizer:
    def __init__(self, mapping):
        self._mapping = mapping

    def decode(self, ids, clean_up_tokenization_spaces=False):
        assert clean_up_tokenization_spaces is False
        return self._mapping[ids[0]]


def test_dedupe_token_ids_by_decoded_string_preserves_first_occurrence():
    tokenizer = _FakeTokenizer(
        {
            0: "a",
            1: "b",
            2: "a",
            3: " <<",
            4: "b",
            5: ">>",
        }
    )

    unique_ids, dropped = _dedupe_token_ids_by_decoded_string(tokenizer, [0, 1, 2, 3, 4, 5])

    assert unique_ids == [0, 1, 3, 5]
    assert dropped == 2


def test_valid_tokens_ids_logits_py_accepts_unique_contiguous_vocab():
    assert _valid_tokens_ids_logits_py(
        ["<<", "x", ">>"],
        [0, 1, 2],
        [0.0, -1e9, 1.0],
    )


def test_valid_tokens_ids_logits_py_rejects_duplicate_tokens():
    assert not _valid_tokens_ids_logits_py(
        ["<<", "x", "x"],
        [0, 1, 2],
        [0.0, 0.0, 0.0],
    )
