# Callgrind profiling results

Workflow: https://github.com/fbreve/pypcc/actions/runs/39483223341
Commit: `88e27a54e663e78393cf23bed5a1705dbc5f330d`
Workload: `benchmarks/profile_kernel_workload.py`, 2048 nodes, 8 classes, 256 particles, 300 iterations; baseline source pinned to `10313ab01207d04ccf3b556bd7701d941b87c020`.

| Callgrind event | Baseline | Fused | Relative difference |
|---|---:|---:|---:|
| Ir (instructions) | 513,533,880 | 504,228,143 | -1.81% |
| Bc (conditional branches) | 78,504,444 | 77,160,582 | -1.71% |
| Bcm (conditional mispredictions, simulated) | 4,636,351 | 4,622,125 | -0.31% |
| Bi (indirect branches) | 9,121,756 | 9,039,943 | -0.90% |
| Bim (indirect mispredictions, simulated) | 2,180,439 | 2,162,573 | -0.82% |

Interpretation: fewer retired instructions and conditional branches under Callgrind are consistent with the Linux speedup from matched wall-clock benchmarking, but are not themselves a speed measurement. Callgrind's branch prediction model is simulated. These totals cover the whole Python process, not solely the Cython kernel. Separate runs compile the variants sequentially; repeated profiling and kernel-function attribution are needed before treating these percentages as precise.

Next: use `callgrind_annotate` to attribute counts to the Cython kernel, and profile Windows/MSVC independently; do not generalize the Linux instruction reduction to Windows.
