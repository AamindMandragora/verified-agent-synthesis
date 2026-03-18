#!/usr/bin/env python3
"""Run eval on a CSD run and print only format_rate and syntax_rate (parse%). Sample size 3 for speed."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "outputs/generated-csd/runs/20260314_222155_2d1042"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "folio"
    # Resolve compiled module
    if (run_dir / "GeneratedCSD.py").exists():
        compiled_path = run_dir
    else:
        for name in ["folio_csd", "gsm_crane_csd", "fol_csd"]:
            d = run_dir / name
            if d.exists() and (d / "GeneratedCSD.py").exists():
                compiled_path = d
                break
        else:
            found = list(run_dir.glob("*/GeneratedCSD.py"))
            compiled_path = found[0].parent if found else run_dir
    from synthesis.evaluator import Evaluator
    ev = Evaluator(
        dataset_name=dataset,
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        device="cuda",
        vocab_size=3000,
        sample_size=3,
        max_steps=256,
        load_in_4bit=True,
    )
    result = ev.evaluate_sample(compiled_module_path=compiled_path, sample_size=3)
    print(f"format_rate={result.format_rate:.2%}  parse(syntax)_rate={result.syntax_rate:.2%}  accuracy={result.accuracy:.2%}  (n={result.num_examples})")

if __name__ == "__main__":
    main()
