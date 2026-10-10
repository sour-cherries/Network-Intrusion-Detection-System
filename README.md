# Network Intrusion Detection (SVD)

This project was built collaboratively as part of the Numerical Analysis course. It implements a network anomaly detection system using Singular Value Decomposition (SVD) written from scratch (without external math libraries for matrix decomposition) and compared against NumPy / LAPACK.

## Team & Contributions

* **Andrei (@Andrei-exe)**: Handled data preparation from end to end : cleaning, one-hot encoding, MinMax scaling, and splitting the dataset into train and test sets.
* **Teodor (@teodor2006728-creator)**: Developed the user interface, error and distribution plots, benchmark comparisons between implementations and performance optimizations.
* **Patrick (@sour-cherries)**: Worked on the core mathematical implementation of the SVD algorithm (power iteration, matrix deflation) and project documentation.

## Project Structure

* `dataPrep.py` - Data cleaning and preprocessing pipeline.
* `svd.py` - Custom SVD implementation and numerical convergence tests.
* `main.py` - End-to-end evaluation script on the test set and metric reporting.
* `rezultat.png` - Reconstruction error visualization plot.

## Evaluation Metrics

Configuring the model to retain k = 5 principal components yielded the following results:
* **Overall Accuracy:** 93.80%
* **Detected Attacks:** 233,611 out of 250,436
* **False Positives:** 2,460

But for futher explanation and details I would advice you to read more on the CN_EN.pdf

## Running the Project

The project does not require external packages; only a standard Python installation is needed.

Run the pipeline:
```bash
python main.py
