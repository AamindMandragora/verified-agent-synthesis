"""The Spider train side must share no example, and no near-duplicate, with the 300 held-out examples.

Spider dev often asks the same SQL twice in different words. The author sees each train
example's gold SQL in its feedback, so a train paraphrase of a held-out question leaks
that held-out answer. Held-out indices are pinned so the baseline numbers stay comparable.
"""
import collections
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLIT = ROOT / "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
HELDOUT_SHA256 = "b588db7134c51941aeca028d26b471cbc23f1f5b066234a80b3c688aae4b025d"
QUESTION_OVERLAP_LIMIT = 0.6


def _load():
    split = json.loads(SPLIT.read_text())
    dev = json.loads((pathlib.Path(split["spider_dir"]) / "dev.json").read_text())
    return split, dev


def _sql_shape(query):
    s = re.sub(r"\s*([(),=<>])\s*", r"\1", query.lower())
    s = re.sub(r"\s+", " ", s).strip().rstrip(";")
    s = re.sub(r"'[^']*'|\"[^\"]*\"", "V", s)
    s = re.sub(r"\b\d+(\.\d+)?\b", "N", s)
    return re.sub(r"\s*order by [^()]*$", "", s)  # a trailing sort doesn't change which rows answer the question


def _words(question):
    return set(re.findall(r"\w+", question.lower()))


def test_heldout_indices_unchanged():
    split, _ = _load()
    assert hashlib.sha256(json.dumps(split["test_indices"]).encode()).hexdigest() == HELDOUT_SHA256


def test_no_shared_index():
    split, _ = _load()
    train, test = split["train_indices"], split["test_indices"]
    assert len(set(train)) == len(train) == 300
    assert not set(train) & set(test)


def test_no_train_example_has_heldout_sql_on_same_db():
    split, dev = _load()
    heldout = {(dev[j]["db_id"], _sql_shape(dev[j]["query"])) for j in split["test_indices"]}
    leaks = [i for i in split["train_indices"] if (dev[i]["db_id"], _sql_shape(dev[i]["query"])) in heldout]
    assert not leaks, leaks


def test_no_train_question_near_duplicates_a_heldout_question_on_same_db():
    split, dev = _load()
    by_db = collections.defaultdict(list)
    for j in split["test_indices"]:
        by_db[dev[j]["db_id"]].append(_words(dev[j]["question"]))
    leaks = []
    for i in split["train_indices"]:
        w = _words(dev[i]["question"])
        if any(len(w & h) / len(w | h) >= QUESTION_OVERLAP_LIMIT for h in by_db[dev[i]["db_id"]]):
            leaks.append(i)
    assert not leaks, leaks
