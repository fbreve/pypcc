"""Summarize exclusive instruction counts from Callgrind's native format."""
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
names = {}
counts = defaultdict(int)
current = None
for raw in path.read_text().splitlines():
    line = raw.strip()
    if line.startswith("fn="):
        value = line[3:]
        if value.startswith("("):
            key, _, rest = value.partition(")")
            key = key[1:]
            if rest.strip():
                names[key] = rest.strip()
            current = names.get(key, value)
        else:
            current = value
    elif line and line[0].isdigit() and current is not None:
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            counts[current] += int(parts[1])
    elif line.startswith(("cfn=", "calls=", "ob=", "fl=", "cob=", "cfi=")):
        pass

total = sum(counts.values())
assert total > 0, "No per-function instruction samples found"
print(f"EXCLUSIVE_INSTRUCTIONS {total}")
print("INSTRUCTIONS PERCENT FUNCTION")
for name, value in sorted(counts.items(), key=lambda item: -item[1])[:60]:
    print(f"{value:12d} {100 * value / total:7.2f}% {name}")
