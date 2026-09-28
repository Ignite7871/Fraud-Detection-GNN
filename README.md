# Fraud Detection with Graph Neural Networks

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
├── frauddetection.ipynb
├── requirements.txt
└── README.md
```

The notebook contains the complete experimental pipeline:

```text
Dataset Loading
      ↓
Feature Preparation
      ↓
Graph Construction
      ↓
Train / Validation / Test Split
      ↓
Logistic Regression Baseline
      ↓
GCN Training
      ↓
Evaluation
      ↓
Confusion Matrix
      ↓
Graph Analysis
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

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the Elliptic Bitcoin Transaction Dataset and place the required files in:

```text
data/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv
```

### 4. Run the notebook

```bash
jupyter notebook frauddetection.ipynb
```

---

## ⚠️ Limitations

This repository is an experimental research project rather than a production fraud-detection system.

Current limitations include:

* random rather than temporal evaluation
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

* temporal train/validation/test splits
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
