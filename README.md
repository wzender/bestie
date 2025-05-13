## 🧠 Bestie — A Lightweight Leaderboard for Text Classification

**Bestie** is a custom-built web app for visualizing and analyzing the performance of text classification models. It is inspired by tools like [Weights & Biases (wandb)](https://wandb.ai), but focused on simplicity and local execution.

The system supports classification into both **types** and **subtypes**. The **type** is automatically inferred using a predefined dictionary mapping (not an LLM), while **subtype** is predicted directly by the model.

---

### 🗉️ Terminology

Following `wandb` conventions:

* **Experiment**: A single execution of a script that trains or evaluates a model.
* **Run**: A single instance of an experiment. Each run corresponds to one record in the leaderboard.
* **Sweep**: A group of runs performed with varying hyperparameters.

---

### 🖥️ UI Layout

The main screen includes two key panels:

1. **Leaderboard Table**

   * Displays all recorded runs.
   * Columns include:

     * `Run ID`
     * `Sweep ID`
     * Model parameters
     * Performance metrics (e.g. accuracy, F1-score, etc.)

2. **Run Analysis**

   * Activated when a specific run is selected.
   * Displays:

     * **Confusion Matrix**: Clickable; clicking a cell reveals misclassified datapoints.
     * **Type & Subtype Histograms**: Visualize distribution of predictions.
     * **Datapoint Table**: Filtered view of individual examples based on user interaction.

---
