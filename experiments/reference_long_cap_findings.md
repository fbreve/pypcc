# Long-cap reference sensitivity (experimental)

Source: GitHub Actions run 41051755488, artifact `reference-long-cap-sensitivity`, commit `f1f0eba`.

Protocol: NumPy experimental PCC backend, no early stopping, 10-NN graph, 2 labeled examples per class, seeds 11/23/37/41/53, four datasets. Each cell below averages five seeds; overall values average 20 runs. The 50,000-iteration prediction is a **comparison endpoint, not verified convergence**.

| Cap | Mean accuracy | Agreement with 50k | Mean wall time (s) |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.67206 | 0.90714 | 1.94708 |
| 20,000 | 0.68878 | 0.94311 | 3.88326 |
| 50,000 | 0.69897 | 1.00000 | 9.63299 |

| Dataset | Accuracy 10k | Accuracy 20k | Accuracy 50k | Agreement 10k vs 50k | Agreement 20k vs 50k |
| --- | ---: | ---: | ---: | ---: | ---: |
| breast_cancer | 0.5274 | 0.5359 | 0.5561 | 0.9161 | 0.9331 |
| digits | 0.5614 | 0.5741 | 0.5900 | 0.8688 | 0.9222 |
| iris | 0.7750 | 0.8125 | 0.7986 | 0.9111 | 0.9694 |
| wine | 0.8244 | 0.8326 | 0.8512 | 0.9326 | 0.9477 |

Interpretation: the 10k cap is not a reliable converged baseline for this protocol. Longer runs generally improve accuracy, but not monotonically on every dataset (Iris 20k exceeds 50k). Agreement is measured only on initially unlabeled examples. Do not infer convergence or superiority of a stopping criterion from these capped comparisons. Before promoting any stopping heuristic, evaluate paired accuracy/elapsed time against a sufficiently characterized reference and multiple label budgets; keep all changes experimental until validated.
