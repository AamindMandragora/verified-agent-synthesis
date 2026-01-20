"""
Metrics tracking for FOLIO evaluation.

Provides a FOLIOMetrics class to track:
- Accuracy (True/False/Uncertain classification)
- FOL structure validity
- Constrained window syntax validity
- Per-label breakdown
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class FOLIOMetrics:
    """Track metrics for FOLIO evaluation."""
    
    # Total counts
    total: int = 0
    correct: int = 0
    
    # Per-label tracking
    true_total: int = 0
    true_correct: int = 0
    false_total: int = 0
    false_correct: int = 0
    uncertain_total: int = 0
    uncertain_correct: int = 0
    
    # Structure validity
    valid_structure: int = 0
    
    # FOL syntax validity (constrained windows)
    total_fol_segments: int = 0
    valid_fol_segments: int = 0
    
    # Predicted label distribution
    predicted_true: int = 0
    predicted_false: int = 0
    predicted_uncertain: int = 0
    predicted_none: int = 0
    
    # Timing
    total_time: float = 0.0
    total_tokens: int = 0
    
    # Individual results for analysis
    results: List[Dict] = field(default_factory=list)
    
    def update(
        self,
        predicted: Optional[str],
        gold: str,
        is_correct: bool,
        valid_structure: bool,
        fol_segments: List[tuple],  # List of (segment_text, is_valid)
        time_seconds: float,
        tokens: int,
        example_id: str = "",
    ):
        """
        Update metrics with a new example.
        
        Args:
            predicted: The predicted answer (True/False/Uncertain/None)
            gold: The gold answer
            is_correct: Whether the prediction matches gold
            valid_structure: Whether the FOL structure is valid
            fol_segments: List of (segment_text, is_valid) tuples
            time_seconds: Time taken for generation
            tokens: Number of tokens generated
            example_id: Identifier for the example
        """
        self.total += 1
        
        if is_correct:
            self.correct += 1
        
        # Per-label tracking
        if gold == "True":
            self.true_total += 1
            if is_correct:
                self.true_correct += 1
        elif gold == "False":
            self.false_total += 1
            if is_correct:
                self.false_correct += 1
        elif gold == "Uncertain":
            self.uncertain_total += 1
            if is_correct:
                self.uncertain_correct += 1
        
        # Structure validity
        if valid_structure:
            self.valid_structure += 1
        
        # FOL segment validity
        for segment_text, is_valid in fol_segments:
            self.total_fol_segments += 1
            if is_valid:
                self.valid_fol_segments += 1
        
        # Predicted label distribution
        if predicted == "True":
            self.predicted_true += 1
        elif predicted == "False":
            self.predicted_false += 1
        elif predicted == "Uncertain":
            self.predicted_uncertain += 1
        else:
            self.predicted_none += 1
        
        # Timing
        self.total_time += time_seconds
        self.total_tokens += tokens
        
        # Store individual result
        self.results.append({
            "example_id": example_id,
            "predicted": predicted,
            "gold": gold,
            "is_correct": is_correct,
            "valid_structure": valid_structure,
            "num_fol_segments": len(fol_segments),
            "valid_fol_segments": sum(1 for _, v in fol_segments if v),
            "time_seconds": time_seconds,
            "tokens": tokens,
        })
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    @property
    def true_accuracy(self) -> float:
        """Accuracy on True labels."""
        return self.true_correct / self.true_total if self.true_total > 0 else 0.0
    
    @property
    def false_accuracy(self) -> float:
        """Accuracy on False labels."""
        return self.false_correct / self.false_total if self.false_total > 0 else 0.0
    
    @property
    def uncertain_accuracy(self) -> float:
        """Accuracy on Uncertain labels."""
        return self.uncertain_correct / self.uncertain_total if self.uncertain_total > 0 else 0.0
    
    @property
    def structure_rate(self) -> float:
        """Rate of valid FOL structure."""
        return self.valid_structure / self.total if self.total > 0 else 0.0
    
    @property
    def syntax_rate(self) -> float:
        """Rate of valid FOL syntax in constrained windows."""
        return self.valid_fol_segments / self.total_fol_segments if self.total_fol_segments > 0 else 0.0
    
    @property
    def avg_time(self) -> float:
        """Average time per example."""
        return self.total_time / self.total if self.total > 0 else 0.0
    
    @property
    def avg_tokens(self) -> float:
        """Average tokens per example."""
        return self.total_tokens / self.total if self.total > 0 else 0.0
    
    def summary(self) -> Dict:
        """Get a summary dict of all metrics."""
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "accuracy_pct": f"{self.accuracy * 100:.1f}%",
            "per_label": {
                "true": {
                    "total": self.true_total,
                    "correct": self.true_correct,
                    "accuracy": self.true_accuracy,
                },
                "false": {
                    "total": self.false_total,
                    "correct": self.false_correct,
                    "accuracy": self.false_accuracy,
                },
                "uncertain": {
                    "total": self.uncertain_total,
                    "correct": self.uncertain_correct,
                    "accuracy": self.uncertain_accuracy,
                },
            },
            "predicted_distribution": {
                "true": self.predicted_true,
                "false": self.predicted_false,
                "uncertain": self.predicted_uncertain,
                "none": self.predicted_none,
            },
            "structure_validity": {
                "valid": self.valid_structure,
                "total": self.total,
                "rate": self.structure_rate,
            },
            "fol_syntax": {
                "valid_segments": self.valid_fol_segments,
                "total_segments": self.total_fol_segments,
                "rate": self.syntax_rate,
            },
            "performance": {
                "total_time": self.total_time,
                "avg_time": self.avg_time,
                "total_tokens": self.total_tokens,
                "avg_tokens": self.avg_tokens,
            },
        }
    
    def print_summary(self):
        """Print a formatted summary of metrics."""
        print("\n" + "=" * 60)
        print("FOLIO Evaluation Results")
        print("=" * 60)
        
        print(f"\nOverall Accuracy: {self.accuracy * 100:.1f}% ({self.correct}/{self.total})")
        
        print("\nPer-Label Accuracy:")
        print(f"  True:      {self.true_accuracy * 100:.1f}% ({self.true_correct}/{self.true_total})")
        print(f"  False:     {self.false_accuracy * 100:.1f}% ({self.false_correct}/{self.false_total})")
        print(f"  Uncertain: {self.uncertain_accuracy * 100:.1f}% ({self.uncertain_correct}/{self.uncertain_total})")
        
        print("\nPredicted Distribution:")
        print(f"  True:      {self.predicted_true}")
        print(f"  False:     {self.predicted_false}")
        print(f"  Uncertain: {self.predicted_uncertain}")
        print(f"  None:      {self.predicted_none}")
        
        print(f"\nStructure Validity: {self.structure_rate * 100:.1f}% ({self.valid_structure}/{self.total})")
        print(f"FOL Syntax Validity: {self.syntax_rate * 100:.1f}% ({self.valid_fol_segments}/{self.total_fol_segments})")
        
        print(f"\nPerformance:")
        print(f"  Total time: {self.total_time:.1f}s")
        print(f"  Avg time/example: {self.avg_time:.2f}s")
        print(f"  Avg tokens/example: {self.avg_tokens:.1f}")
        
        print("=" * 60)
    
    def to_json(self, filepath: str):
        """Save metrics to JSON file."""
        data = self.summary()
        data["individual_results"] = self.results
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> "FOLIOMetrics":
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics = cls()
        metrics.total = data["total"]
        metrics.correct = data["correct"]
        metrics.results = data.get("individual_results", [])
        
        # Reconstruct other fields from results if available
        for result in metrics.results:
            gold = result["gold"]
            if gold == "True":
                metrics.true_total += 1
                if result["is_correct"]:
                    metrics.true_correct += 1
            elif gold == "False":
                metrics.false_total += 1
                if result["is_correct"]:
                    metrics.false_correct += 1
            elif gold == "Uncertain":
                metrics.uncertain_total += 1
                if result["is_correct"]:
                    metrics.uncertain_correct += 1
            
            if result.get("valid_structure"):
                metrics.valid_structure += 1
            
            metrics.total_fol_segments += result.get("num_fol_segments", 0)
            metrics.valid_fol_segments += result.get("valid_fol_segments", 0)
            
            predicted = result.get("predicted")
            if predicted == "True":
                metrics.predicted_true += 1
            elif predicted == "False":
                metrics.predicted_false += 1
            elif predicted == "Uncertain":
                metrics.predicted_uncertain += 1
            else:
                metrics.predicted_none += 1
            
            metrics.total_time += result.get("time_seconds", 0)
            metrics.total_tokens += result.get("tokens", 0)
        
        return metrics
