"""
Dafny verification wrapper for CSD synthesis.

Runs `dafny verify` on generated code and parses the results.
"""

import os
import re
import subprocess
import tempfile


def _dafny_subprocess_env() -> dict:
    """Return env for dafny subprocesses with /opt/anaconda/bin prepended to PATH.

    Dafny 4.x requires a `z3` binary discoverable on PATH (or next to the dafny
    binary). Our install has no bundled z3; /opt/anaconda/bin/z3 (4.12.2) works.
    Login shells see it, but nohup/systemd-style parent processes often don't.
    """
    env = os.environ.copy()
    extra = "/opt/anaconda/bin"
    current = env.get("PATH", "")
    if extra not in current.split(":"):
        env["PATH"] = f"{extra}:{current}" if current else extra
    return env
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VerificationDiagnostic:
    """Structured summary of one verifier finding for refinement prompts."""

    file: str
    line: int
    column: int
    message: str
    obligation_kind: str
    failing_text: str = ""
    source_excerpt: str = ""
    call_name: str = ""
    related_file: Optional[str] = None
    related_line: Optional[int] = None
    related_message: str = ""
    contract_excerpt: str = ""


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
    diagnostics: list[VerificationDiagnostic] = field(default_factory=list)
    
    def get_error_summary(self) -> str:
        """Get a human-readable summary of errors.

        If raw Dafny output is available, return it directly — it includes source
        snippets and Related location lines that identify the exact failing contract.
        """
        if self.success:
            return "Verification successful"

        if self.raw_output:
            return self.raw_output.strip()

        if not self.errors:
            return self.raw_stderr or "Unknown verification failure"

        lines = [f"Verification failed with {len(self.errors)} error(s):"]
        for err in self.errors:
            lines.append(f"  - Line {err.line}: {err.message}")
        return "\n".join(lines)

    def get_structured_feedback(self) -> str:
        """Return a compact structured verifier summary for refinement prompts."""
        if self.success or not self.diagnostics:
            return ""

        lines = ["Structured verification analysis:"]
        for idx, diagnostic in enumerate(self.diagnostics, start=1):
            location = f"{Path(diagnostic.file).name}:{diagnostic.line}"
            lines.append(f"{idx}. {diagnostic.obligation_kind.title()} failure at {location}")
            lines.append(f"   Message: {diagnostic.message}")
            if diagnostic.call_name:
                lines.append(f"   Related call: {diagnostic.call_name}(...)")
            if diagnostic.failing_text:
                lines.append(f"   Failing code: {diagnostic.failing_text}")
            if diagnostic.related_file and diagnostic.related_line:
                related_location = f"{Path(diagnostic.related_file).name}:{diagnostic.related_line}"
                related_message = diagnostic.related_message or "Related contract location from Dafny"
                lines.append(f"   Related contract: {related_location} ({related_message})")
            if diagnostic.source_excerpt:
                lines.append("   Local code excerpt:")
                lines.extend(f"     {line}" for line in diagnostic.source_excerpt.splitlines())
            if diagnostic.contract_excerpt:
                lines.append("   Relevant contract excerpt:")
                lines.extend(f"     {line}" for line in diagnostic.contract_excerpt.splitlines())

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
    RELATED_PATTERN = re.compile(
        r"^(.+?)\((\d+),(\d+)\):\s*Related location:\s*(.+)$"
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

    @staticmethod
    def _classify_obligation(message: str) -> str:
        lowered = message.lower()
        if "loop invariant" in lowered or "invariant could not be proved" in lowered:
            return "invariant"
        if "precondition for this call" in lowered:
            return "precondition"
        if "postcondition could not be proved" in lowered:
            return "postcondition"
        if "decreases expression" in lowered:
            return "decreases"
        return "verification"

    @staticmethod
    def _extract_line(text: str, line_no: int) -> str:
        lines = text.splitlines()
        if 1 <= line_no <= len(lines):
            return lines[line_no - 1].strip()
        return ""

    @staticmethod
    def _extract_excerpt(text: str, line_no: int, radius: int = 2) -> str:
        lines = text.splitlines()
        if not lines or line_no <= 0:
            return ""
        start = max(1, line_no - radius)
        end = min(len(lines), line_no + radius)
        excerpt_lines = []
        for current in range(start, end + 1):
            marker = ">" if current == line_no else " "
            excerpt_lines.append(f"{marker} {current}: {lines[current - 1]}")
        return "\n".join(excerpt_lines)

    @staticmethod
    def _extract_call_name(source_line: str) -> str:
        match = re.search(r"(?:helpers\.|lm\.|parser\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", source_line)
        return match.group(1) if match else ""

    def _match_error_blocks(self, output: str, errors: list[VerificationError]) -> list[list[str]]:
        lines = output.splitlines()
        error_indices = [idx for idx, line in enumerate(lines) if self.ERROR_PATTERN.match(line)]
        blocks: list[list[str]] = []
        for pos, _ in enumerate(errors):
            if pos >= len(error_indices):
                blocks.append([])
                continue
            start = error_indices[pos]
            end = error_indices[pos + 1] if pos + 1 < len(error_indices) else len(lines)
            blocks.append(lines[start:end])
        return blocks

    def _build_diagnostics(
        self,
        output: str,
        errors: list[VerificationError],
        generated_source: str,
        proof_source: str,
    ) -> list[VerificationDiagnostic]:
        diagnostics: list[VerificationDiagnostic] = []
        blocks = self._match_error_blocks(output, errors)

        for error, block in zip(errors, blocks):
            source_line = self._extract_line(generated_source, error.line)
            diagnostic = VerificationDiagnostic(
                file=error.file,
                line=error.line,
                column=error.column,
                message=error.message,
                obligation_kind=self._classify_obligation(error.message),
                failing_text=source_line,
                source_excerpt=self._extract_excerpt(generated_source, error.line),
                call_name=self._extract_call_name(source_line),
            )

            for raw_line in block:
                related = self.RELATED_PATTERN.match(raw_line)
                if not related:
                    continue
                diagnostic.related_file = related.group(1)
                diagnostic.related_line = int(related.group(2))
                diagnostic.related_message = related.group(4).strip()
                related_name = Path(diagnostic.related_file).name
                if related_name == "VerifiedAgentSynthesis.dfy":
                    diagnostic.contract_excerpt = self._extract_excerpt(proof_source, diagnostic.related_line)
                elif related_name == "GeneratedCSD.dfy":
                    diagnostic.contract_excerpt = self._extract_excerpt(generated_source, diagnostic.related_line)
                break

            diagnostics.append(diagnostic)

        return diagnostics
    
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
            
            proof_source_text = ""
            if source_proof.exists():
                proof_source_text = source_proof.read_text()
                (temp_path / "VerifiedAgentSynthesis.dfy").write_text(
                    proof_source_text
                )
                
                # Create agents directory for relative imports
                agents_dir = temp_path / "agents"
                agents_dir.mkdir(exist_ok=True)
                (agents_dir / "VerifiedAgentSynthesis.dfy").write_text(
                    proof_source_text
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
                    cwd=temp_path,  # Run from temp dir so includes work
                    env=_dafny_subprocess_env()
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
            diagnostics = self._build_diagnostics(
                combined_output,
                errors,
                dafny_code,
                proof_source_text,
            )
            
            # Dafny returns 0 on success
            success = result.returncode == 0 and len(errors) == 0
            
            return VerificationResult(
                success=success,
                errors=errors,
                raw_output=result.stdout,
                raw_stderr=result.stderr,
                return_code=result.returncode,
                diagnostics=diagnostics
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

