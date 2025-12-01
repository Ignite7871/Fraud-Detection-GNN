📌 How to Execute the Project
1️⃣ Download the Dataset

This project uses the Elliptic Bitcoin Transaction Dataset.
You can download it from Kaggle:

👉 Search: “Elliptic Bitcoin dataset Kaggle”

After downloading, extract the following CSV files:
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv

Place them inside the project folder like:
fraud-detection-gnn/
└── data/
    ├── elliptic_txs_features.csv
    ├── elliptiptic_txs_classes.csv
    └── elliptic_txs_edgelist.csv

pip install -r requirements.txt

Run the Notebook

What the Code Does
✔ Loads the Elliptic dataset (Bitcoin transactions)
✔ Builds a graph where:
Nodes = transactions
Edges = money flow between transactions
✔ Trains a Logistic Regression baseline
✔ Builds and trains a Graph Convolutional Network (GCN)
✔ Evaluates with:
Accuracy
F1-score
ROC-AUC score
Confusion Matrix
✔ Shows how graph learning helps identify fraud more accurately


You should see results like:
Accuracy: 0.9459
F1 Score: 0.6515
ROC-AUC: 0.9448

