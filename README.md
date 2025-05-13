## 🧠 Magellan  — A Lightweight Leaderboard for Text Classification

**Magellan** is a custom-built web app for visualizing and analyzing the performance of text classification models. It is inspired by tools like [Weights & Biases (wandb)](https://wandb.ai), but focused on simplicity and local execution.

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

Multiple page view with the following pages (each page replace the other, there can be other pages in the future)
2.1. Error Analysis Page:
   * On loading displays the analysis for the first row
   * On selecting rows the analysis changed for the selected row
   * Displays:
     * **Type Histograms**: Clickable bar plot representing the value counts of the true_type
     * **Subtype confusion matrix**: Clickable confusion matrix, using ff.create_annotated_heatmap with labels of count and percentage
     * **Datapoint Table**: Filtered view of individual examples based on user clicks on the Subtype confusion matrix.
2.2. Full Run Table:
   * Data table that will display the:
   - text (text)
   - true type (true_type)
   - true sub type (true_sub_type)
   - predicated sub type (pred_sub_type)
   - correct sub type : boolean of true_sub_type==pred_sub_type (correct)

The design should be Lightweight and modern and based on dash bootstrap lux
---
