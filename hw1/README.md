# HW1: Traditional Information Retrieval Methods

In this assignment, we implement following methods:

- Vector Space Model (TF-IDF)
- Probabilic Models:
    - Binary Independence Model (BIM)
    - Okapi BM25

And we evaluate the methods using:
- P@K
- P@10
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)


## Note:
I have tried my best to harness the power of NumPy and do calculations using matrix multiplication which increases the speed dramatically. 

The current performance bottleneck is processing the docment texts which can be fixed by preprocessing the text and saving it.