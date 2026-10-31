# Sampled late prediction changes (40k–50k)

Source: GitHub Actions run 41142792672, artifact `late-prediction-changes/late-flip-sequences.csv`, commit `8fcd530`.

Protocol: NumPy experimental PCC backend without early stopping; four datasets, five seeds (11, 23, 37, 41, 53), two labeled examples per class, 10-NN graph. Independently replayed checkpoints at 40k, 42k, 44k, 46k, 48k and 50k iterations, using identical seeds and initial labels. Counts aggregate across 20 dataset/seed runs, so a node is counted separately for each run.

| Dataset | Unlabeled | Nodes changed at least once | Nodes changed > once | Adjacent changes | Changed then returned to initial | Different at endpoints |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iris | 720 | 2 | 0 | 2 | 0 | 2 |
| wine | 860 | 21 | 6 | 27 | 6 | 15 |
| breast_cancer | 2825 | 128 | 37 | 169 | 33 | 95 |
| digits | 8885 | 169 | 3 | 172 | 3 | 166 |
| **Total** | **13290** | **320** | **46** | **370** | **42** | **278** |

The endpoint comparison misses 42 nodes (13.1% of the 320 nodes with at least one sampled change) that returned to their original predicted class. Repeated sampled changes concentrate in breast_cancer (37/46). These snapshots are not a complete per-iteration trajectory; they cannot determine how many additional flips occurred between checkpoints or prove an oscillatory dynamical regime. A stable endpoint or large dominance margin alone is insufficient to establish convergence. Do not promote a stopping criterion based on this evidence alone.
