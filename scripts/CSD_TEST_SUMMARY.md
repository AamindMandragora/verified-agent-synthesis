# CSD test summary: FOLIO and GSM (format% & parse%)

**Date:** 2026-03-15  
**Branch:** advayth/attempted_library_fixes  
**Eval:** synthesis evaluator, 3 samples, Qwen2.5-Coder-3B-Instruct (4-bit), `scripts/quick_eval_metrics.py`.

---

## Can the pipeline make a CSD for both?

**Yes.** Existing runs with compiled modules:

- **FOLIO:** `outputs/generated-csd/runs/20260314_222155_2d1042/folio_csd`
- **GSM:**   `outputs/generated-csd/runs/20260315_033138_44e52c/gsm_crane_csd`

---

## Format% and parse% (syntax_rate)

| Dataset   | format_rate | parse% (syntax_rate) | accuracy (n=3) |
|-----------|-------------|----------------------|----------------|
| **FOLIO** | **100%**    | **100%**             | 0%             |
| **GSM**   | **100%**    | **0%**               | 0%             |

- **FOLIO:** Outputs consistently have valid `$ ... %` structure and the formula segment parses under the FOL (Prover9) grammar. Content is often shallow (e.g. `Alkale(mix)`, `P1( be)`), so accuracy is 0%, but format and parse are fine.
- **GSM:** Outputs have valid `<< ... >>` structure (format 100%), but the extracted expressions often fail the arithmetic grammar (e.g. no variables, or repeated tokens like `100% 100% ...`, `195 - 15 - 15 - ...`), so parse% is 0%.

---

## How to re-run the quick metrics

```bash
# FOLIO (3 samples)
python scripts/quick_eval_metrics.py outputs/generated-csd/runs/20260314_222155_2d1042 folio

# GSM (3 samples)
python scripts/quick_eval_metrics.py outputs/generated-csd/runs/20260315_033138_44e52c gsm_symbolic
```

Full eval with more samples and detailed output:

```bash
python scripts/eval_csd_with_synthesis_prompt.py --run-dir <run_dir> --dataset folio|gsm_symbolic --sample-size 8 --4bit
```

---

## Conclusion

- **FOLIO CSD:** Somewhat decent for format and parse (100% / 100% on 3 examples). Accuracy can be improved later (model content, prompt, or tuning).
- **GSM CSD:** Decent format (100%); parse is poor (0%) because the model often does not produce grammar-valid arithmetic with variables. Improving the GSM strategy or prompt to encourage variables and valid expressions would help parse%.
