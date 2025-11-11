# KNN / LDA / QDA Combined Model

## Tested only with the basic ~30 data samples

Different models make different mistakes:

- KNN misclassified circle → diagonal_left
- LDA misclassified diagonal_left → circle
- QDA misclassified diagonal_right → diagonal_left

These mistakes do not overlap. So during the vote: 

| Model | Circle sample prediction |
| ----- | ------------------------ |
| KNN   | wrong                    |
| LDA   | correct                  |
| QDA   | correct                  |

→ Majority wins → final prediction correct.

This is exactly why ensembles exist.
Multiple weak learners → one strong classifier.

Final Interpretation:

The extracted features successfully separate shapes based on their geometric orientation. Circle samples show balanced values, horizontal shapes show extremely low x₃, and vertical shapes show extremely high x₃. Diagonal shapes fall between these extremes. Individually, KNN, LDA, and QDA achieve ~0.89 accuracy due to occasional class overlap, especially between circle and diagonal_left. However, because each model makes different mistakes, a voting-based ensemble achieves 100% accuracy, confirming that the models complement each other and that ensemble learning is the optimal approach for this dataset.