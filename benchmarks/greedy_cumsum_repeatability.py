"""Run one independent A/B benchmark trial and print its verified summary.

Each GitHub Actions matrix job uses a separate fresh runner. The underlying
benchmark alternates implementation order and validates exact state equality.
"""
import json
import subprocess
import sys
from pathlib import Path

trial = int(sys.argv[1])
output = Path(f"greedy-repeat-{trial}.jsonl")
with output.open("w", encoding="utf-8") as stream:
    subprocess.run(
        [sys.executable, "benchmarks/greedy_cumsum_ab.py"],
        stdout=stream, check=True,
    )
lines = output.read_text(encoding="utf-8").splitlines()
rows = [json.loads(line) for line in lines if line.startswith("{")]
summaries = [json.loads(line[len("SUMMARY "):]) for line in lines if line.startswith("SUMMARY ")]
if len(rows) != 108 or len(summaries) != 1 or summaries[0].get("cases") != 108:
    raise RuntimeError(f"Unexpected benchmark output: {len(rows)} rows, {len(summaries)} summaries")
print("REPEATABILITY_SUMMARY " + json.dumps({"trial": trial, **summaries[0]}), flush=True)
