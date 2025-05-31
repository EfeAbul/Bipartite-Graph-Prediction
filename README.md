
# Bipartite Graph Classification using Graph Partitioning and Neural Networks

This project explores the classification of bipartite graphs using deep learning methods. It combines classical graph theory techniques (like the Kernighan–Lin partitioning algorithm) with convolutional neural networks (CNNs) trained on adjacency matrix representations of graphs.

---

## Project Overview

- **Goal**: Automatically determine whether a given graph is bipartite based on its structure.
- **Approach**:
  - Generate synthetic graphs using various random graph models.
  - Label graphs heuristically using the Kernighan–Lin algorithm (approximating bipartiteness).
  - Represent graphs as 2D adjacency matrices.
  - Train a CNN on the matrices to classify graphs as bipartite or not.

---

##  Files

- `Bipartite_Graph.ipynb`: Main notebook containing data generation, labeling, model training, and evaluation.

---

##  Graph Generation

The dataset consists of graphs generated using the following models:
- **Erdős–Rényi (ER)**: Random edge assignment.
- **Barabási–Albert (BA)**: Scale-free graphs with preferential attachment.
- **Watts–Strogatz (WS)**: Small-world graphs.
- **Random Regular Graphs**: Graphs where all nodes have the same degree.

Each graph is converted into a fixed-size adjacency matrix (e.g., 20×20) suitable for input into a CNN.

---

##  Model Architecture

A convolutional neural network (CNN) is used to classify the adjacency matrix:

```text
Input → Conv2D → Flatten → Dense → Output
```

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Final layer uses `sigmoid` activation to output bipartite probability

---

##  Labeling Method

Unlike traditional approaches that use `networkx.is_bipartite()`, this project:
- Uses the **Kernighan–Lin algorithm** to partition the graph into two sets
- Uses this partitioning as a **proxy label** for bipartiteness
- This allows the model to learn from structural approximations of bipartite separation

---

##  Results & Insights

After training, the model is evaluated on a separate test set to measure its ability to classify unseen graphs. Evaluation metrics include:
- Accuracy
- Loss
- Sample prediction outputs

Visualization of training performance and example graph structures is included in the notebook.

---

##  How to Run

1. Clone the repository and open the notebook:
   ```bash
   jupyter notebook Bipartite_Graph.ipynb
   ```

2. Install required packages if not already available:
   ```bash
   pip install networkx numpy pandas matplotlib tensorflow
   ```

3. Run all cells to generate the dataset, train the model, and see predictions.

---

##  Requirements

This project uses:
- Python 3.x
- TensorFlow / Keras
- NetworkX
- Matplotlib
- NumPy
- Pandas

---

## ✍️ Author

This project was developed as part of an academic exercise on graph classification and machine learning.
