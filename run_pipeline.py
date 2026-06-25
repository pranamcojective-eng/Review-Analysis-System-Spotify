#!/usr/bin/env python3
"""CLI entry point to run the Review Discovery Pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_pipeline


def main():
    print("=" * 60)
    print("  Review Discovery Engine - Pipeline Runner")
    print("=" * 60)
    result = run_pipeline()
    print()
    print(f"  Raw reviews:      {result.raw_count}")
    print(f"  Cleaned unique:   {result.cleaned_count}")
    print(f"  Duplicates removed: {result.duplicates_removed}")
    print(f"  Theme clusters:   {len(result.clusters)}")
    print()
    print("  Key Insights:")
    for insight in result.report.key_insights_summary:
        print(f"    • {insight}")
    print()
    print("  Output saved to ./output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
