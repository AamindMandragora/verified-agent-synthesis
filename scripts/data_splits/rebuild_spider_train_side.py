"""Rebuild the Spider train side: drop examples that leak a held-out answer, refill, rebalance.

Held-out (test_indices) is never touched. A train example leaks when, on the same database,
its gold SQL matches a held-out gold SQL once literals and a trailing ORDER BY are masked, or its question shares at
least 60% of its words with a held-out question. Spider dev asks many queries twice in
different words, and the author sees train gold SQL, so such pairs leak held-out answers.
Each dropped example is replaced by an unused dev example of the same difficulty level
(seeded shuffle), then train is reordered so every 50-block matches the held-out mix.
Usage: python scripts/data_splits/rebuild_spider_train_side.py
"""
import collections
import json
import pathlib
import random
import re

SPLIT = pathlib.Path(__file__).resolve().parents[2] / "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
LEVELS = ("easy", "medium", "hard", "extra")
BLOCK = 50
QUESTION_OVERLAP_LIMIT = 0.6


def sql_shape(query):
    s = re.sub(r"\s*([(),=<>])\s*", r"\1", query.lower())
    s = re.sub(r"\s+", " ", s).strip().rstrip(";")
    s = re.sub(r"'[^']*'|\"[^\"]*\"", "V", s)
    s = re.sub(r"\b\d+(\.\d+)?\b", "N", s)
    return re.sub(r"\s*order by [^()]*$", "", s)  # a trailing sort doesn't change which rows answer the question


def words(question):
    return set(re.findall(r"\w+", question.lower()))


def leak_checker(dev, test):
    heldout_sql = {(dev[j]["db_id"], sql_shape(dev[j]["query"])) for j in test}
    heldout_words = collections.defaultdict(list)
    for j in test:
        heldout_words[dev[j]["db_id"]].append(words(dev[j]["question"]))

    def leaks(i):
        db, w = dev[i]["db_id"], words(dev[i]["question"])
        if (db, sql_shape(dev[i]["query"])) in heldout_sql:
            return True
        return any(len(w & h) / len(w | h) >= QUESTION_OVERLAP_LIMIT for h in heldout_words[db])

    return leaks


def refill(train, test, dev, level_of, seed):
    leaks = leak_checker(dev, test)
    kept = [i for i in train if not leaks(i)]
    dropped = [i for i in train if leaks(i)]
    taken_sql = {(dev[i]["db_id"], sql_shape(dev[i]["query"])) for i in kept}
    used = set(train) | set(test)
    pool = {lv: [i for i in range(len(dev)) if i not in used and level_of[str(i)] == lv and not leaks(i)] for lv in LEVELS}
    rng = random.Random(seed)
    for lv in LEVELS:
        rng.shuffle(pool[lv])
    added = []
    for i in dropped:
        lv = level_of[str(i)]
        # Prefer a query train doesn't already have; Spider dev repeats many queries, so fall back
        # to a repeat (harmless: it only repeats train, never held-out). An empty pool raises.
        fresh = [c for c in pool[lv] if (dev[c]["db_id"], sql_shape(dev[c]["query"])) not in taken_sql]
        c = fresh[-1] if fresh else pool[lv][-1]
        pool[lv].remove(c)
        taken_sql.add((dev[c]["db_id"], sql_shape(dev[c]["query"])))
        added.append(c)
    return kept + added, dropped, added


def balanced_order(train, level_of):
    queues = {lv: collections.deque(i for i in train if level_of[str(i)] == lv) for lv in LEVELS}
    totals = {lv: len(q) for lv, q in queues.items()}
    n_blocks = len(train) // BLOCK
    order, taken = [], {lv: 0 for lv in LEVELS}
    for b in range(1, n_blocks + 1):
        # Cumulative quota rounding keeps each block within one example of the exact share.
        quota = {lv: round(totals[lv] * b / n_blocks) - taken[lv] for lv in LEVELS}
        short = BLOCK - sum(quota.values())
        for lv in sorted(LEVELS, key=lambda lv: totals[lv] * b / n_blocks - taken[lv] - quota[lv], reverse=True)[:max(short, 0)]:
            quota[lv] += 1
        block = [queues[lv].popleft() for lv in LEVELS for _ in range(quota[lv])]
        for lv in LEVELS:
            taken[lv] += quota[lv]
        order += sorted(block, key=train.index)
    assert sorted(order) == sorted(train)
    return order


def main():
    split = json.loads(SPLIT.read_text())
    dev = json.loads((pathlib.Path(split["spider_dir"]) / "dev.json").read_text())
    level_of = split["difficulty_by_index"]
    before = list(split["train_indices"])
    train, dropped, added = refill(before, split["test_indices"], dev, level_of, split["seed"])
    split["train_indices"] = balanced_order(train, level_of)
    split["train_composition"] = dict(collections.Counter(level_of[str(i)] for i in train))
    split["train_order"] = "blocks of 50 balanced to the held-out difficulty mix"
    split["train_leak_filter"] = {
        "date": "2026-09-25",
        "rule": "drop train examples that, on the same db, match a held-out gold SQL (literals and trailing ORDER BY masked) "
                f"or share >= {QUESTION_OVERLAP_LIMIT:.0%} of question words with a held-out question; "
                "refill from unused dev examples of the same difficulty (seeded)",
        "dropped": dropped,
        "added": added,
    }
    SPLIT.write_text(json.dumps(split, indent=2) + "\n")
    print(f"dropped {len(dropped)} leaking train examples, added {len(added)}; first 10 train={split['train_indices'][:10]}")


if __name__ == "__main__":
    main()
