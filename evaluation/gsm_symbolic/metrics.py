"""
Evaluation metrics for GSM-Symbolic.

Provides a dataclass for tracking and computing evaluation metrics
across multiple examples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GSMMetrics:
    """
    Metrics for GSM-Symbolic evaluation.
    
    Tracks:
    - Answer accuracy (exact numeric match)
    - Valid format rate (outputs contain proper delimiters)
    - Syntax validity (math expressions pass grammar validation)
    - Token usage and timing
    """
    n: int = 0
    correct: int = 0
    valid_format: int = 0
    syntax_valid_segments: int = 0
    total_segments: int = 0
    total_tokens: int = 0
    total_time: float = 0.0

    def accuracy(self) -> float:
        """Compute answer accuracy as percentage."""
        return 100.0 * self.correct / max(1, self.n)

    def format_rate(self) -> float:
        """Compute valid format rate as percentage."""
        return 100.0 * self.valid_format / max(1, self.n)

    def syntax_validity(self) -> float:
        """Compute syntax validity rate as percentage."""
        return 100.0 * self.syntax_valid_segments / max(1, self.total_segments)

    def avg_tokens(self) -> float:
        """Compute average tokens per example."""
        return self.total_tokens / max(1, self.n)

    def avg_time(self) -> float:
        """Compute average time per example in seconds."""
        return self.total_time / max(1, self.n)
    
    def update(
        self,
        is_correct: bool,
        is_valid_format: bool,
        token_count: int,
        time_seconds: float,
        constrained_segments: list = None
    ) -> None:
        """
        Update metrics with a single example result.
        
        Args:
            is_correct: Whether the answer was correct
            is_valid_format: Whether the output format was valid
            token_count: Number of tokens generated
            time_seconds: Time taken for generation
            constrained_segments: Optional list of (segment_text, is_valid) tuples
        """
        self.n += 1
        self.correct += 1 if is_correct else 0
        self.valid_format += 1 if is_valid_format else 0
        self.total_tokens += token_count
        self.total_time += time_seconds
        
        if constrained_segments:
            for seg_text, is_valid in constrained_segments:
                self.total_segments += 1
                if is_valid:
                    self.syntax_valid_segments += 1
    
    def summary(self) -> str:
        """Generate a summary string of the metrics."""
        lines = [
            f"Examples: {self.n}",
            f"Answer Accuracy: {self.accuracy():.1f}%",
            f"Valid Format Rate: {self.format_rate():.1f}%",
        ]
        if self.total_segments > 0:
            lines.append(f"Syntax Validity: {self.syntax_validity():.1f}% ({self.syntax_valid_segments}/{self.total_segments} segments)")
        lines.extend([
            f"Avg Tokens: {self.avg_tokens():.1f}",
            f"Avg Time: {self.avg_time():.2f}s",
        ])
        return "\n".join(lines)
