"""Re-score saved SMILES baseline/metadecode JSONs on the new metric:
unique-valid rate (= unique_valid_count / N), Tanimoto diversity, RDKit validity.

Recomputes smiles_eval per answer from the stored `generated_answer` string +
the class grammar + prompt exemplars (loaded from the repo) — no re-generation.

Usage: python rescore_smiles_unique_valid.py <dir> <class>
  e.g. python rescore_smiles_unique_valid.py outputs/controlled_comparison/smiles_7B/chain_extenders chain_extenders
Writes nothing; prints a table. (Re-baselining is read-only; we set bars from this.)
"""
import json, sys, glob, os
from synthesis.evaluate.benchmarks.smiles.dataset import load_smiles
from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output, smiles_trial_metrics

def main(d, cls):
    ex = load_smiles(classes=[cls], samples_per_class=1, split_name="test", trial=1)[0]  # grammar/exemplars only; split-independent
    grammar_text = ex.get("grammar_text", "")
    exemplars = ex.get("prompt_exemplars", [])
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        name = os.path.basename(p)
        if name.startswith("cars_allclass"):
            continue
        d_ = json.load(open(p))
        answers = d_.get("answers", [])
        if not answers:
            continue
        samples = []
        for a in answers:
            out = a.get("generated_answer", "")
            se = evaluate_smiles_output(cls, out, grammar_text, exemplars, require_rdkit=True)
            samples.append({"smiles_eval": se})
        n = len(samples)
        trial = smiles_trial_metrics(samples)
        uv = trial["unique_valid_count"]
        rows.append((
            name.replace(".json", ""),
            uv / max(1, n),                       # unique-valid RATE (new accuracy)
            trial["validity_rdkit"],              # validity (comparable axis)
            trial["diversity_tanimoto"],          # diversity (comparable axis)
            uv, n,
            d_.get("accuracy"),                   # OLD stored accuracy (membership rate)
        ))
    print(f"\n=== {d}  (class={cls}) ===")
    print(f"{'strategy':16} {'uniqValRate':>11} {'validity':>9} {'diversity':>9} {'uniq/N':>8} {'OLDacc':>7}")
    for nm, uvr, val, div, uv, n, old in rows:
        divs = f"{div:.3f}" if div is not None else "n/a"
        print(f"{nm:16} {uvr:>11.3f} {val:>9.3f} {divs:>9} {str(uv)+'/'+str(n):>8} {old:>7}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
