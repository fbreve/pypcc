"""Summarize exclusive instruction counts from a single-event Callgrind profile."""
import re
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("Usage: summarize_callgrind.py PROFILE")

lines = Path(sys.argv[1]).read_text().splitlines()
events = None
expected = None
names = {}
counts = defaultdict(int)
current = None
pending_call_cost = False
position_re = re.compile(r"^(?:\d+|[+\-*]\d*|\*)$")
for raw in lines:
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("events:"):
        events = line.partition(":")[2].split()
        if events != ["Ir"]:
            raise ValueError(f"Expected only Ir event, got {events!r}")
    elif line.startswith("totals:"):
        if expected is not None:
            raise ValueError("Multiple totals records")
        expected = int(line.partition(":")[2].strip())
    elif line.startswith("fn="):
        value = line[3:]
        match = re.fullmatch(r"\((\d+)\)(?:\s+(.*))?", value)
        if match:
            key, name = match.groups()
            if name:
                names[key] = name
            current = names.get(key, f"fn-id:{key}")
        else:
            current = value
        pending_call_cost = False
    elif line.startswith("calls="):
        pending_call_cost = True
    elif line.startswith(("cfn=", "cob=", "cfi=", "fl=", "fi=", "fe=", "ob=")):
        continue
    elif current is not None:
        parts = line.split()
        if len(parts) >= 2 and position_re.fullmatch(parts[0]) and parts[1].isdigit():
            if pending_call_cost:
                pending_call_cost = False
            else:
                counts[current] += int(parts[1])

if events != ["Ir"] or expected is None or expected <= 0:
    raise ValueError("Missing valid Ir event and positive totals")
total = sum(counts.values())
if total != expected:
    raise ValueError(f"Exclusive instruction mismatch: summed={total}, totals={expected}")
print(f"EXCLUSIVE_INSTRUCTIONS {total}")
print("INSTRUCTIONS PERCENT FUNCTION")
for name, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:60]:
    print(f"{value:12d} {100 * value / total:7.2f}% {name}")
