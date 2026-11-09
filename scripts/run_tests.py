"""Automated Test Suite Runner (Phase 10).

Executes unit tests and evaluation tests sequentially, reporting pass/fail status.
"""

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent


def main():
    print("=" * 65)
    print("PHASE 10: AUTOMATED TEST SUITE RUNNER")
    print("=" * 65)

    test_suites = [
        "tests/unit",
        "tests/evaluation",
    ]

    for suite in test_suites:
        print(f"\n---> Running suite: {suite} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", suite, "-v"],
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            print(f"\nFAILED: Test suite '{suite}' had failures.")
            sys.exit(result.returncode)

    print("\n" + "=" * 65)
    print("ALL SELECTED TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()

