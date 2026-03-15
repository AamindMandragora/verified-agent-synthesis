# FOLIO: Ideal Answers vs CSD Output (Attempt 2 & 8)

## Ideal answer format

For each example we generate an answer that:

1. **Starts with reasoning** (unconstrained text): e.g. "From the given premises we can represent the conclusion in first-order logic. The statement to evaluate corresponds to the following formula."
2. **Puts one FOL formula in delimiters**: ` << Alkane(mixture) {and} Contain(mixture, carbon) >> `
3. **Ends with the required sentence**: "The final statement to evaluate is therefore << *statement* >>" where *statement* is the **gold conclusion** in grammar form (`{and}`, `{exists}`, etc.).

Extraction uses the same regex as the evaluator: `<<\s*([^<>]+?)\s*>>`. The **last** segment is sent to Prover9 as the conclusion, with gold premises from the dataset.

## Prover9 verification of ideal answers

| Example | Expected | Ideal last << >> segment | Prover9 result | Match |
|--------|----------|---------------------------|----------------|-------|
| 1 | False | `Alkane(mixture) {and} Contain(mixture, carbon)` | *parse_fail* | N/A (premises 5–6 use equality/XOR; FOL parser rejects) |
| 2 | Uncertain | `SecondLargestChineseCity(beijing)` | Unknown | Yes |
| 3 | Uncertain | `{exists}x (FootballClub(x) {and} LoanedTo(ailtonSilva, x))` | Unknown | Yes |

So once the last constrained part is extracted, it **is** equivalent to the correct output for examples 2 and 3 (Prover9 returns Unknown ≡ Uncertain). Example 1’s ideal conclusion is correct, but we could not run Prover9 because some **gold premises** fail to parse (equality `y=z` and XOR in the FOL pipeline).

---

## Diff: Ideal vs CSD output

### Example 1 (expected False)

| | Ideal | Attempt 2 | Attempt 8 |
|---|--------|-----------|-----------|
| **Last << >> segment** | `Alkane(mixture) {and} Contain(mixture, carbon)` | Prose + repeated `{and}` (not one formula) | `Alkale(mix)` |
| **Why wrong** | — | Model put explanatory text and malformed “formulas” inside << >> | Typo (Alkale), wrong constant (mix), missing second conjunct |

So we get **negative matches**: the CSD never outputs the gold conclusion; it either outputs prose inside << >> or a wrong formula. Prover9 then fails to parse or would return the wrong label.

### Example 2 (expected Uncertain)

| | Ideal | Attempt 2 | Attempt 8 |
|---|--------|-----------|-----------|
| **Last << >> segment** | `SecondLargestChineseCity(beijing)` | Prose (“first order form of ‘be’…”) | `SecondLargest( be  )` |
| **Prover9** | Unknown ✓ | parse_fail | Unknown ✓ |
| **Graded** | — | Wrong | **Correct** |

So we get a **false positive** in attempt 8: the CSD output is the **wrong formula** (predicate `SecondLargest` with odd spacing instead of `SecondLargestChineseCity`), but Prover9 still returns Unknown (cannot prove or disprove), which normalizes to Uncertain and is counted as correct.

### Example 3 (expected Uncertain)

| | Ideal | Attempt 2 | Attempt 8 |
|---|--------|-----------|-----------|
| **Last << >> segment** | `{exists}x (FootballClub(x) {and} LoanedTo(ailtonSilva, x))` | `{!!!!!...}` | `{!!!!!...}` |
| **Why wrong** | — | Garbage inside << >>; not valid FOL | Same; syntax invalid, Prover9 parse_fail |

**Negative matches**: the model produced invalid content between << >>, so we never get a verifiable conclusion.

---

## Why we get negative matches or false positives

1. **Negative matches (should be correct but marked wrong)**  
   - **Ex 1:** CSD does not output the gold conclusion (prose or wrong formula like `Alkale(mix)`).  
   - **Ex 3:** CSD outputs non-FOL garbage (`{!!!!...}`) between << >>.  
   So the **last constrained part** is either unparseable or wrong → Prover9 fails or gives the wrong label.

2. **False positive (only in attempt 8, Ex 2)**  
   - The **last constrained part** is `SecondLargest( be  )`, which is **not** the gold conclusion `SecondLargestChineseCity(beijing)`.  
   - Prover9 still returns **Unknown** (it can’t prove or disprove this formula from the premises).  
   - Unknown is normalized to Uncertain, which matches the expected label, so the example is counted **correct** even though the formula is wrong.

3. **Root cause**  
   - **Format:** Ideal format (reasoning, then one << FOL >>, then “the final statement to evaluate is therefore << statement >>”) is not what the CSD produces; it often puts prose or multiple fragments inside << >> or outputs wrong/garbage formulas.  
   - **Grading:** We only compare the **Prover9 result** (True/False/Unknown) to the expected label. We do **not** check that the conclusion formula equals the gold. So a wrong but “Unknown-yielding” formula (Ex 2) can still be a match.

To reduce false positives, we could add a check that the **last << >> segment** is syntactically valid FOL and optionally that it matches or is equivalent to the gold conclusion, and only then use the Prover9 result for grading.
