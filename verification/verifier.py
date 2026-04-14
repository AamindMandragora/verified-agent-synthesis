"""
Verification wrapper for Python-first CSD synthesis.

Transpiles generated Python strategy code to Dafny, runs `dafny verify`,
and parses the results.
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from verification.dafny_runner import check_dafny_available, prepare_temp_dafny_dir


@dataclass
class VerificationError:
    """A single verification error from Dafny."""
    file: str
    line: int
    column: int
    message: str
    error_type: str = "Error"
    
    def __str__(self) -> str:
        return f"{self.file}({self.line},{self.column}): {self.error_type}: {self.message}"


@dataclass
class VerificationResult:
    """Result of running Dafny verification."""
    success: bool
    errors: list[VerificationError] = field(default_factory=list)
    raw_output: str = ""
    raw_stderr: str = ""
    return_code: int = 0
    
    def get_error_summary(self) -> str:
        """Get a human-readable summary of errors for the LLM refinement prompt."""
        if self.success:
            return "Verification successful"
        
        combined_raw = (self.raw_output or "") + "\n" + (self.raw_stderr or "")
        combined_raw = combined_raw.strip()

        if not self.errors:
            # No parsed errors, return raw output so the model still sees Dafny's message
            return combined_raw or "Verification failed (no details captured)."

        lines = [f"Dafny verification failed with {len(self.errors)} error(s):", ""]
        for err in self.errors:
            # Include line, column, and full message so the model can fix the exact location
            lines.append(f"  (Line {err.line}, Column {err.column}): {err.error_type}: {err.message}")
        lines.append("")
        # Append raw Dafny output so the model sees the exact errors (e.g. multi-line, related hints)
        if combined_raw:
            raw_preview = combined_raw if len(combined_raw) <= 3500 else combined_raw[:3500] + "\n... (truncated)"
            lines.append("Full Dafny output:")
            lines.append(raw_preview)
        return "\n".join(lines)


class DafnyVerifier:
    """
    Wrapper for Dafny verification.
    
    Writes generated Python code to a temp workspace, transpiles it to Dafny,
    runs verification, and parses results.
    """

    # Regex patterns for parsing Dafny output
    ERROR_PATTERN = re.compile(
        r"^(.+?)\((\d+),(\d+)\):\s*(Error|Warning|Info):\s*(.+)$",
        re.MULTILINE
    )
    
    def __init__(
        self,
        dafny_path: str = "dafny",
        timeout: int = 60,
        extra_args: Optional[list[str]] = None
    ):
        """
        Initialize the verifier.
        
        Args:
            dafny_path: Path to dafny executable
            timeout: Verification timeout in seconds
            extra_args: Additional arguments to pass to dafny
        """
        self.dafny_path = dafny_path
        self.timeout = timeout
        self.extra_args = extra_args or []
        
        # Verify dafny is available
        check_dafny_available(self.dafny_path)

    def _parse_errors(self, output: str, source_file: str) -> list[VerificationError]:
        """
        Parse verification errors from Dafny output.
        
        Args:
            output: Raw Dafny output
            source_file: Path to the source file being verified
            
        Returns:
            List of parsed errors
        """
        errors = []
        
        for match in self.ERROR_PATTERN.finditer(output):
            file_path = match.group(1)
            line = int(match.group(2))
            column = int(match.group(3))
            error_type = match.group(4)
            message = match.group(5).strip()
            
            # Only include actual errors (not warnings/info)
            if error_type == "Error":
                errors.append(VerificationError(
                    file=file_path,
                    line=line,
                    column=column,
                    message=message,
                    error_type=error_type
                ))
        
        return errors
    
    def verify(self, python_code: str) -> VerificationResult:
        """
        Verify generated Python strategy code.
        
        Args:
            python_code: Complete Python source code
            
        Returns:
            VerificationResult with success status and any errors
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            try:
                source_file, cwd, _ = prepare_temp_dafny_dir(temp_path, python_code)
            except Exception as e:
                return VerificationResult(
                    success=False,
                    errors=[VerificationError(
                        file="System", line=0, column=0,
                        message=str(e),
                    )],
                    return_code=-1,
                )

            # Run dafny verify
            cmd = [
                self.dafny_path,
                "verify",
                str(source_file),
                *self.extra_args
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=cwd,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    success=False,
                    errors=[VerificationError(
                        file=str(source_file),
                        line=0,
                        column=0,
                        message=f"Verification timed out after {self.timeout} seconds"
                    )],
                    raw_output="",
                    raw_stderr="Timeout",
                    return_code=-1
                )
            
            # Parse the output
            combined_output = result.stdout + result.stderr
            errors = self._parse_errors(combined_output, str(source_file))
            
            # Dafny returns 0 on success
            success = result.returncode == 0 and len(errors) == 0
            
            return VerificationResult(
                success=success,
                errors=errors,
                raw_output=result.stdout,
                raw_stderr=result.stderr,
                return_code=result.returncode
            )
    
    def verify_file(self, file_path: Path) -> VerificationResult:
        """
        Verify a generated Python strategy file directly.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            VerificationResult
        """
        if not file_path.exists():
            return VerificationResult(
                success=False,
                errors=[VerificationError(
                    file=str(file_path),
                    line=0,
                    column=0,
                    message=f"File not found: {file_path}"
                )],
                return_code=-1
            )
        
        return self.verify(file_path.read_text())
