# Fraud Detection with Graph Neural Networks

[![Tests](https://github.com/Ignite7871/Fraud-Detection-GNN/actions/workflows/tests.yml/badge.svg)](https://github.com/Ignite7871/Fraud-Detection-GNN/actions/workflows/tests.yml)

A graph-based fraud detection experiment using the **Elliptic Bitcoin Transaction Dataset** to investigate whether transaction relationships provide useful information beyond transaction-level features.

The project compares a traditional **Logistic Regression baseline** against a two-layer **Graph Convolutional Network (GCN)** using transaction features and transaction-flow relationships.

---

## 🔍 Problem

Cryptocurrency transactions can be represented as a graph:

```text
Transaction ───→ Transaction
     │                 │
     └──── money flow ─┘
```

A transaction can therefore be described using both:

1. its own feature vector
2. its relationships with other transactions

This project investigates whether graph-based message passing can exploit that relational structure for illicit-transaction detection.

---

## 🧠 Approach

### Baseline

A Logistic Regression classifier is trained using the transaction-level feature vectors.

### Graph Neural Network

A two-layer Graph Convolutional Network is constructed using:

* transaction features as node attributes
* transaction relationships as graph edges
* weighted cross-entropy to address class imbalance

The GCN architecture is:

```text
Node Features
      ↓
   GCNConv
      ↓
    ReLU
      ↓
   Dropout
      ↓
   GCNConv
      ↓
 Class Prediction
```

---

## 📊 Experimental Setup

### Dataset

The project uses the **Elliptic Bitcoin Transaction Dataset**.

The dataset is processed into:

```text
Nodes  → Bitcoin transactions
Edges  → Transaction relationships
Labels → Licit / Illicit / Unknown
```

Unknown labels are excluded from supervised evaluation.

### Data split

Labeled transactions are divided into:

* 64% training
* 16% validation
* 20% test

using stratified random splitting with `random_state=42`.

---

## 📈 Results

### Test Set

| Model               |   Accuracy |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: |
| Logistic Regression | **0.9621** | **0.7946** | **0.9680** |
| GCN                 |     0.8797 |     0.5900 |     0.9525 |


### Interpretation

On the current experimental split, the Logistic Regression baseline performs better than the GCN across the reported test metrics.

This means the current implementation does **not** establish that graph convolution improves fraud detection performance.

Instead, the experiment demonstrates an important baseline comparison: strong transaction-level features can provide substantial predictive performance, while the current GCN configuration does not yet outperform that baseline.

---

## 🛡️ Class-Imbalance Analysis

The test-set confusion matrix for the GCN is:

```text
                 Predicted
               Licit  Illicit
Actual Licit    7387     1017
Actual Illicit   103     806
```

The resulting GCN metrics are:

```text
Illicit precision: 0.4421
Illicit recall:    0.8867
Illicit F1:        0.5900
```

The model therefore identifies a large proportion of illicit transactions, but at the cost of a relatively high false-positive rate.

This trade-off is especially important in fraud detection, where missed illicit activity and unnecessary investigation alerts have different operational costs.

---

## 🔬 Graph Analysis

The notebook also measures the proportion of fraudulent neighbors connected to test transactions.

Current experiment:

```text
Average fraudulent-neighbor ratio: 0.0323
```

This provides a simple view of how illicit transactions are distributed within the transaction graph.

---

## ⏱️ Temporal Evaluation

In addition to the random stratified split, the project evaluates the models using chronological transaction time steps.

The temporal experiment uses:

```text
Earlier time steps
        ↓
     Training

Middle time steps
        ↓
    Validation

Later time steps
        ↓
      Test
```

The split is performed at the time-step level so that a single time step cannot appear in multiple partitions.

For the GCN, the training graph is restricted to transactions occurring at or before the training cutoff. Validation and test evaluation use graph views restricted to their respective chronological cutoffs, preventing later-period graph edges from being used during earlier evaluation.

The Logistic Regression baseline uses the same chronological node partitions without graph information.

The notebook reports both random-split and temporal-split results so the effect of chronological evaluation can be examined directly.

### Evaluation note

The temporal experiment should be interpreted as a chronological holdout protocol rather than a complete simulation of production real-time fraud scoring. It evaluates generalization to later time periods while retaining the benchmark's available transaction features.

### Temporal split

Using a 60/20/20 chronological split over the labeled time steps:

| Partition  | Time steps | Nodes |
| ---------- | ---------: | ----: |
| Train      |       1–29 | 26381 |
| Validation |      30–39 |  8999 |
| Test       |      40–49 | 11184 |

### Results (temporal test set)

| Model               |   Accuracy |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: |
| Logistic Regression | **0.7101** | **0.2303** | **0.8308** |
| GCN                 |     0.6124 |     0.1997 |     0.8111 |

The test-set confusion matrix for the temporal GCN is:

```text
                 Predicted
               Licit  Illicit
Actual Licit    6308     4240
Actual Illicit    95      541
```

### Random vs. temporal comparison

| Model               | Random ROC-AUC | Temporal ROC-AUC | Random F1 | Temporal F1 |
| -------------------- | --------------: | -----------------: | ---------: | ------------: |
| Logistic Regression  |          0.9680 |              0.8308 |     0.7946 |        0.2303 |
| GCN                  |          0.9525 |              0.8111 |     0.5900 |        0.1997 |

### Interpretation

Both models degrade substantially under chronological evaluation. ROC-AUC drops by roughly 0.14–0.15 points for each model, and F1 drops much more sharply (Logistic Regression: 0.7946 → 0.2303; GCN: 0.5900 → 0.1997), driven mainly by a large increase in false positives on illicit transactions in the later time steps.

Logistic Regression still has a higher ROC-AUC than the GCN under the temporal protocol, so the random-split conclusion — that this GCN configuration does not outperform the transaction-level baseline — is not overturned by moving to a stricter chronological split. Both models' absolute performance is markedly weaker than the random-split numbers suggest, which is itself a useful finding: it indicates the random split was likely optimistic relative to how these models would generalize to later, unseen time periods.

---

## 🧩 Ablation Study

To determine where the predictive signal originates, the project evaluates four configurations under both the random and temporal protocols:

| Configuration | Description |
| --- | --- |
| Features only | Logistic Regression using transaction-level features |
| Graph only | GCN using constant node features, forcing the model to rely on graph structure and message passing |
| Features + GCN | Original two-layer GCN using transaction-level features and graph connectivity |
| Features + neighborhood features | Logistic Regression using transaction features augmented with one-hop mean-neighbor features |

The graph-only configuration keeps the GCN architecture unchanged but replaces all transaction features with a constant scalar for every node. This isolates the contribution of graph connectivity and message passing.

The neighborhood-feature configuration provides a simpler alternative to learned message passing: each transaction receives the mean feature vector of its one-hop neighbors, concatenated with its original feature vector. No transaction labels are used to construct these features.

All four configurations are evaluated using the same train/validation/test partitions within each protocol.

### Random-Split Ablation Results

| Configuration                     |   Accuracy |         F1 |    ROC-AUC |
| ---------------------------------- | ---------: | ---------: | ---------: |
| Features only                      |     0.9621 |     0.7946 |     0.9680 |
| Graph only                         |     0.8219 |     0.0768 |     0.5938 |
| Features + GCN                     |     0.8797 |     0.5900 |     0.9525 |
| Features + neighborhood features   | **0.9688** | **0.8283** | **0.9758** |

### Temporal-Split Ablation Results

| Configuration                     |   Accuracy |         F1 |    ROC-AUC |
| ---------------------------------- | ---------: | ---------: | ---------: |
| Features only                      |     0.7101 |     0.2303 |     0.8308 |
| Graph only                         |     0.7729 |     0.0790 |     0.5750 |
| Features + GCN                     |     0.6124 |     0.1997 |     0.8111 |
| Features + neighborhood features   | **0.8131** | **0.2881** | **0.8323** |

Each split's neighborhood features are computed from that split's own chronological graph snapshot (training-cutoff features for training rows, validation-cutoff for validation rows, test-cutoff for test rows), rather than reusing the training-cutoff snapshot for every row.

### Random vs. temporal ablation summary

| Configuration                     | Random ROC-AUC | Temporal ROC-AUC | Random F1 | Temporal F1 |
| ---------------------------------- | --------------: | -----------------: | ---------: | ------------: |
| Features only                      |          0.9680 |              0.8308 |     0.7946 |        0.2303 |
| Graph only                         |          0.5938 |              0.5750 |     0.0768 |        0.0790 |
| Features + GCN                     |          0.9525 |              0.8111 |     0.5900 |        0.1997 |
| Features + neighborhood features   |          0.9758 |              0.8323 |     0.8283 |        0.2881 |

### Interpretation

Graph topology alone carries almost no predictive signal for this task: the graph-only GCN (constant node features, so predictions can only come from connectivity and message passing) performs close to chance in both protocols (ROC-AUC 0.59 random-split, 0.58 temporal-split). Removing the graph entirely and keeping only transaction features (Logistic Regression) reaches 0.97 / 0.83 ROC-AUC — dramatically higher.

Augmenting the transaction features with a simple one-hop mean-neighbor feature and feeding them to Logistic Regression — no message passing, no learned graph model — matches or exceeds the GCN's performance in both the random split (ROC-AUC 0.976 vs 0.953, F1 0.828 vs 0.590) and the temporal split (ROC-AUC 0.832 vs 0.811, F1 0.288 vs 0.200). Under the temporal protocol, this neighborhood-augmented Logistic Regression is in fact the strongest of all four configurations on ROC-AUC, narrowly ahead of the features-only baseline (0.8323 vs 0.8308).

Taken together, this ablation indicates that in this configuration, almost all of the GCN's discriminative power comes from the transaction-level node features it has access to rather than from its message-passing mechanism, and that a much simpler explicit neighborhood-averaging feature captures at least as much of the locally available relational signal as the learned GCN does. This does not rule out that a different or better-tuned GNN architecture could do better — that question is left for future work — but the current results do not establish an advantage for graph-based message passing over the transaction-level and simple-neighborhood-aggregate baselines on this dataset.

---

## 🧪 Evaluation Metrics

The project reports:

* Accuracy
* F1 Score
* ROC-AUC
* Confusion Matrix
* Precision
* Recall

ROC-AUC is included because class imbalance makes raw accuracy alone insufficient for evaluating a fraud-detection classifier.

---

## 🏗️ Project Structure

```text
Fraud-Detection-GNN/
│
├── src/
│   ├── data.py
│   ├── models.py
│   ├── baselines.py
│   ├── train.py
│   ├── evaluation.py
│   └── utils.py
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── frauddetection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

Dataset loading, feature preparation, graph construction, the train/validation/test split, the GCN model, training loop, and evaluation all live in `src/` as reusable, tested functions. The notebook is the experiment/demo layer that calls into them:

```text
Environment
    ↓
Imports + Seed
    ↓
Prepare Dataset
    ↓
Run Baseline
    ↓
Train GCN
    ↓
Evaluate
    ↓
Analyze
    ↓
Visualize
```

---

## ⚙️ Technologies

![Python](https://img.shields.io/badge/Python-3776AB?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-3C2179?logo=pytorch\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy\&logoColor=white)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ignite7871/Fraud-Detection-GNN.git
cd Fraud-Detection-GNN
```

### 2. Create a Python 3.12 environment and install dependencies

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name fraud-gnn --display-name "Fraud GNN (Python 3.12)"
```

### 4. Download the dataset

Download the Elliptic Bitcoin Transaction Dataset and place the required files in:

```text
data/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv
```

### 5. Run the notebook

```bash
jupyter notebook frauddetection.ipynb
```

Select the **Fraud GNN (Python 3.12)** kernel before running.

---

## 🧪 Testing

`src/` has a `tests/` suite covering data preparation, the GCN model, and the training loop, independent of the 660 MB dataset. CI runs this suite on every push and pull request via GitHub Actions.

```bash
pytest -q
```

---

## ⚠️ Limitations

This repository is an experimental research project rather than a production fraud-detection system.

Current limitations include:

* temporal evaluation is performed at the dataset time-step level rather than exact transaction timestamps
* benchmark features may contain aggregated neighborhood information, so this is not a full point-in-time production simulation
* neighborhood-feature ablations are transductive within each graph snapshot because unlabeled node features from connected evaluation nodes may contribute to neighborhood aggregates
* a relatively simple two-layer GCN
* heuristic graph construction from the available edge list
* no systematic hyperparameter search
* no threshold optimisation for operational fraud-detection costs
* no comparison with more advanced graph models
* no repeated-seed statistical analysis

The current results should therefore be interpreted as a baseline experiment rather than evidence of production-level fraud detection performance.

---

## 🔭 Future Work

Potential extensions include:

* GraphSAGE and GAT comparisons
* class-imbalance strategies beyond weighted loss
* threshold tuning based on precision/recall trade-offs
* repeated runs with multiple random seeds
* feature ablation studies
* neighborhood-feature analysis
* calibration analysis
* explainability for individual transaction predictions

---

## 👤 Author

**Srikar Reddy Gunupati**

B.Tech — Computer Science & Engineering (AI & ML)

Research interests:

**Machine Learning • Graph Neural Networks • AI Security • Fraud Detection • Intelligent Systems**

[GitHub](https://github.com/Ignite7871)

[LinkedIn](https://linkedin.com/in/srikar-reddy-gunupati)
