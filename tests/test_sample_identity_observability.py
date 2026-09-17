from synthesis.evaluate.evaluator import EvaluationResult


def test_sample_identity_metadata_for_gsm_and_spider_examples():
    gsm_sample = EvaluationResult._sample_identity_metadata(
        {
            "crane_source_index": 37,
            "id_orig": 12,
            "id_shuffled": 98,
        },
        4,
    )
    assert gsm_sample["example_index"] == 4
    assert gsm_sample["crane_source_index"] == 37
    assert gsm_sample["source_index"] == 37
    assert gsm_sample["id_orig"] == 12
    assert gsm_sample["id_shuffled"] == 98

    spider_sample = EvaluationResult._sample_identity_metadata(
        {
            "spider_source_index": 18,
            "db_id": "concert_singer",
        },
        9,
    )
    assert spider_sample["example_index"] == 9
    assert spider_sample["spider_source_index"] == 18
    assert spider_sample["source_index"] == 18
    assert spider_sample["db_id"] == "concert_singer"


def test_sample_observability_annotation_keeps_identity_fields():
    sample = {
        "question": "dummy question",
        "question_full": "dummy question",
        "expected": "1",
        "actual": "2",
        "full_output": "answer",
        "scored_output": "answer",
        "answer_source": "last_visible_span",
        "has_extracted_answer": True,
        "is_correct": False,
        "accuracy_applicable": True,
        "contains_delimiters": True,
        "visible_delimiters": True,
        "used_constrained_chunk": False,
        "is_syntax_valid": False,
        "syntax_rate": 0.0,
        "num_visible_spans": 0,
        "num_valid_visible_spans": 0,
        "visible_span_token_lengths": [],
        "valid_visible_span_token_lengths": [],
        "token_count": 0,
        "hit_max_steps": False,
        "time_seconds": 1.0,
        "runtime_budget_exceeded": False,
        "helper_trace": [],
        "task_guidance": [],
        "example_index": 5,
        "db_id": "concert_singer",
        "spider_source_index": 18,
        "source_index": 18,
    }

    annotated = EvaluationResult._annotate_sample_observability(sample)

    assert annotated["example_index"] == 5
    assert annotated["db_id"] == "concert_singer"
    assert annotated["spider_source_index"] == 18
    assert annotated["source_index"] == 18
    assert annotated["failure_location"] == "span_absent"
    assert annotated["answer_provenance"]
    assert annotated["provenance_tags"]
