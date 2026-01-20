"""
Dafny verification wrapper for CSD synthesis.

Runs `dafny verify` on generated code and parses the results.
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
        """Get a human-readable summary of errors."""
        if self.success:
            return "Verification successful"
        
        if not self.errors:
            # No parsed errors, return raw output
            return self.raw_output or self.raw_stderr
        
        lines = [f"Verification failed with {len(self.errors)} error(s):"]
        for err in self.errors:
            lines.append(f"  - Line {err.line}: {err.message}")
        return "\n".join(lines)


class DafnyVerifier:
    """
    Wrapper for Dafny verification.
    
    Writes Dafny code to a temp file, runs verification, and parses results.
    """
    
    # Path to proofs directory (for includes)
    PROOFS_DIR = Path(__file__).parent.parent / "proofs"
    
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
        self._check_dafny_available()
    
    def _check_dafny_available(self) -> None:
        """Check that Dafny is installed and accessible."""
        try:
            result = subprocess.run(
                [self.dafny_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Dafny returned non-zero exit code: {result.stderr}"
                )
        except FileNotFoundError:
            raise RuntimeError(
                f"Dafny not found at '{self.dafny_path}'. "
                "Please install Dafny and ensure it's in your PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Dafny version check timed out")
    
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
    
    def verify(self, dafny_code: str) -> VerificationResult:
        """
        Verify Dafny code.
        
        Args:
            dafny_code: Complete Dafny source code
            
        Returns:
            VerificationResult with success status and any errors
        """
        # Create temp directory to hold the file
        # We need to be in a directory where the include paths work
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # The VerifiedAgentSynthesis.dfy and GeneratedCSD.dfy are expected
            # to be in the same directory by default Dafny include semantics unless paths are relative
            # The template says: include "VerifiedAgentSynthesis.dfy"
            
            # Copy the VerifiedAgentSynthesis.dfy to temp location
            source_proof = self.PROOFS_DIR / "VerifiedAgentSynthesis.dfy"
            # Fallback: check dafny/ directory if not in proofs/
            if not source_proof.exists():
                source_proof = Path(__file__).parent.parent / "dafny" / "VerifiedAgentSynthesis.dfy"
            
            if source_proof.exists():
                (temp_path / "VerifiedAgentSynthesis.dfy").write_text(
                    source_proof.read_text()
                )
                
                # Create agents directory for relative imports
                agents_dir = temp_path / "agents"
                agents_dir.mkdir(exist_ok=True)
                (agents_dir / "VerifiedAgentSynthesis.dfy").write_text(
                    source_proof.read_text()
                )
                
            else:
                return VerificationResult(
                    success=False,
                    errors=[VerificationError(
                        file="System", line=0, column=0,
                        message=f"VerifiedAgentSynthesis.dfy not found in proofs/ or dafny/"
                    )],
                    return_code=-1
                )
            
            # Write the generated code
            source_file = temp_path / "GeneratedCSD.dfy"
            source_file.write_text(dafny_code)
            
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
                    cwd=temp_path  # Run from temp dir so includes work
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
        Verify a Dafny file directly.
        
        Args:
            file_path: Path to the Dafny file
            
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

