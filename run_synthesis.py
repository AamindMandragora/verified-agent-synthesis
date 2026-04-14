#!/usr/bin/env python3
"""
Compatibility wrapper for the main synthesis CLI.

The implementation now lives in `synthesis/cli/run_synthesis.py`.
"""

from synthesis.cli.run_synthesis import main


if __name__ == "__main__":
    raise SystemExit(main())
