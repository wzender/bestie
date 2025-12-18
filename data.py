import pandas as pd
import numpy as np
from uuid import uuid4

# Define types and subtypes
all_types = [chr(65 + i) for i in range(20)] + ["Perfect", "WorstMin", "Medium"]
np.random.seed(42)
all_subtypes = []
for t in all_types:
    num_subtypes = np.random.randint(10, 16)
    all_subtypes.extend([f"{t}{i}" for i in range(1, num_subtypes + 1)])
all_unk_subtypes = [f"UNK{i}" for i in range(1, 11)] + [None]

def generate_mock_data():
    np.random.seed(42)

    # Generate shared datapoints that will be used for all benchmarks
    rows_per_benchmark = 300
    shared_texts = [f"Sample text {i}" for i in range(rows_per_benchmark)]

    # Generate true labels for shared datapoints (same for all runs)
    type_to_subtypes = {
        t: [s for s in all_subtypes if s.startswith(t)] for t in all_types
    }

    shared_true_types = np.random.choice(all_types, rows_per_benchmark)
    shared_true_subtypes = []
    for t in shared_true_types:
        shared_true_subtypes.append(np.random.choice(type_to_subtypes[t]))

    # Define benchmarks
    n_benchmarks = 5
    benchmarks = [f"subtype_00_{i:02d}" for i in range(n_benchmarks)]

    # Create runs - multiple runs per benchmark
    n_runs_per_benchmark = 2
    run_data_list = []
    run_number = 1

    for bench_idx, benchmark in enumerate(benchmarks):
        for run_offset in range(n_runs_per_benchmark):
            run_data_list.append({
                "run_number": run_number,
                "run_id": str(uuid4())[:8],
                "sweep_id": f"sweep_{bench_idx % 3}",
                "model_name": f"llama-{(bench_idx + run_offset) % 3 + 1}",
                "benchmark": benchmark,
                "model_params": f"lr:0.{bench_idx % 5}, layers:{(bench_idx % 3) + 1}",
                "accuracy": np.random.uniform(0.7, 0.95),
                "f1_score": np.random.uniform(0.65, 0.9),
            })
            run_number += 1

    run_data = pd.DataFrame(run_data_list)

    # Generate detailed data - same datapoints for all runs
    detailed_data_list = []

    for _, run_row in run_data.iterrows():
        run_id = run_row["run_id"]

        # Generate predictions for this run (different predictions per run, same texts)
        np.random.seed(hash(run_id) % (2**32))  # Deterministic but different per run

        pred_subtypes = []
        for true_type in shared_true_types:
            # Each run has slight variation in predictions
            if np.random.random() < 0.85:  # 85% accuracy baseline
                # Correct prediction
                pred_subtypes.append(np.random.choice(type_to_subtypes[true_type]))
            else:
                # Incorrect prediction - pick a different type
                other_types = [t for t in all_types if t != true_type]
                wrong_type = np.random.choice(other_types)
                pred_subtypes.append(np.random.choice(type_to_subtypes[wrong_type]))

        run_data_chunk = pd.DataFrame({
            "run_id": run_id,
            "text": shared_texts,
            "true_type": shared_true_types,
            "true_subtype": shared_true_subtypes,
            "pred_subtype": pred_subtypes,
        })

        run_data_chunk["correct"] = (
            run_data_chunk["true_subtype"] == run_data_chunk["pred_subtype"]
        )

        # Simulated model confidence between 0 and 1
        np.random.seed(44 + hash(run_id) % (2**32))
        run_data_chunk["confidence"] = np.random.uniform(0, 1, size=len(run_data_chunk))

        np.random.seed(43 + hash(run_id) % (2**32))
        run_data_chunk["pred_unk_subtype"] = np.random.choice(
            all_unk_subtypes, size=len(run_data_chunk), p=[0.09] * 10 + [0.1]
        )

        detailed_data_list.append(run_data_chunk)

    detailed_data = pd.concat(detailed_data_list, ignore_index=True)

    # Update accuracy and f1_score in run_data based on actual predictions
    for idx, run_id in enumerate(run_data["run_id"]):
        run_predictions = detailed_data[detailed_data["run_id"] == run_id]
        accuracy = run_predictions["correct"].mean()

        # Calculate F1 score per subtype and average
        f1_scores = []
        for true_subtype in run_predictions["true_subtype"].unique():
            mask = run_predictions["true_subtype"] == true_subtype
            if mask.sum() > 0:
                tp = ((run_predictions["pred_subtype"] == true_subtype) & mask).sum()
                fp = ((run_predictions["pred_subtype"] == true_subtype) & ~mask).sum()
                fn = mask.sum() - tp
                if tp + fp + fn > 0:
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        f1_scores.append(f1)

        f1_score_avg = np.mean(f1_scores) if f1_scores else 0.5

        run_data.loc[idx, "accuracy"] = accuracy
        run_data.loc[idx, "f1_score"] = f1_score_avg

    unique_benchmarks = sorted(run_data["benchmark"].unique())
    test_run_id = run_data.iloc[0]["run_id"]

    return run_data, detailed_data, test_run_id, unique_benchmarks

# Create benchmark options
benchmark_options = [{"label": bm, "value": bm} for bm in generate_mock_data()[3]]
