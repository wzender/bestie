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
    n_samples = 10
    run_data = pd.DataFrame(
        {
            "run_id": [str(uuid4())[:8] for _ in range(n_samples)],
            "sweep_id": [f"sweep_{i%3}" for i in range(n_samples)],
            "model_name": [f"llama-{i%3 + 1}" for i in range(n_samples)],
            "benchmark": [f"subtype_00_{i%5:02d}" for i in range(n_samples)],
            "model_params": [f"lr:0.{i%5}, layers:{(i%3)+1}" for i in range(n_samples)],
            "accuracy": np.random.uniform(0.7, 0.95, n_samples),
            "f1_score": np.random.uniform(0.65, 0.9, n_samples),
        }
    )
    test_run_data = pd.DataFrame(
        {
            "Text": [f"Text{i+1}" for i in range(20)],
            "true_type": ["Perfect"] * 5 + ["WorstMin"] * 5 + ["Medium"] * 10,
            "true_subtype": [f"Perfect{i}" for i in range(1, 6)]
            + [f"WorstMin{i}" for i in range(1, 6)]
            + [f"Medium{i}" for i in range(1, 6)]
            + [f"Medium{i}" for i in range(6, 11)],
            "pred_subtype": [f"Perfect{i}" for i in range(1, 6)]
            + ["Else"] * 3
            + ["Else", f"WorstMin5"]
            + [f"Medium{i}" for i in range(1, 5)]
            + [f"Medium6"]
            + [f"Medium{i}" for i in range(7, 10)]
            + [f"Medium{i}" for i in range(1, 3)],
            "correct": [True] * 5
            + [False] * 3
            + [False, True]
            + [True] * 4
            + [False]
            + [True] * 3
            + [False] * 2,
        }
    )
    test_run_id = "test_run"
    test_run_data["run_id"] = test_run_id
    test_run_data["pred_subtype"] = test_run_data.apply(
        lambda row: (
            row["pred_subtype"]
            if row["pred_subtype"] != "Else"
            else np.random.choice(
                [s for s in all_subtypes if s.startswith(row["true_type"])]
            )
        ),
        axis=1,
    )
    test_run_data["correct"] = (
        test_run_data["true_subtype"] == test_run_data["pred_subtype"]
    )
    np.random.seed(43)
    test_run_data["pred_unk_subtype"] = np.random.choice(
        all_unk_subtypes, size=len(test_run_data), p=[0.09] * 10 + [0.1]
    )
    test_run_summary = pd.DataFrame(
        {
            "run_id": [test_run_id],
            "sweep_id": ["test_sweep"],
            "model_name": ["llama-1"],
            "benchmark": ["subtype_00_01"],
            "model_params": ["test_params"],
            "accuracy": [test_run_data["correct"].mean()],
            "f1_score": [0.65],
        }
    )
    run_data = pd.concat([run_data, test_run_summary], ignore_index=True)
    n_detailed = 1000
    detailed_data = []
    rows_per_run = 500
    type_to_subtypes = {
        t: [s for s in all_subtypes if s.startswith(t)] for t in all_types
    }
    for run_idx, run_id in enumerate(run_data["run_id"][:-1]):
        num_types = np.random.randint(3, 12)
        run_types = np.random.choice(all_types, num_types, replace=False)
        run_subtypes_by_type = {}
        for t in run_types:
            available_subtypes = type_to_subtypes[t]
            num_subtypes_per_type = np.random.randint(10, 20)
            selected_subtypes = np.random.choice(
                available_subtypes,
                min(num_subtypes_per_type, len(available_subtypes)),
                replace=False,
            )
            run_subtypes_by_type[t] = list(selected_subtypes)
        run_subtypes = []
        for t in run_types:
            run_subtypes.extend(run_subtypes_by_type[t])
        if len(run_subtypes) > 20:
            final_subtypes = []
            for t in run_types:
                final_subtypes.append(np.random.choice(run_subtypes_by_type[t]))
            remaining_slots = 20 - len(final_subtypes)
            remaining_subtypes = [s for s in run_subtypes if s not in final_subtypes]
            if remaining_subtypes and remaining_slots > 0:
                additional_subtypes = np.random.choice(
                    remaining_subtypes,
                    min(remaining_slots, len(remaining_subtypes)),
                    replace=False,
                )
                final_subtypes.extend(additional_subtypes)
            run_subtypes = final_subtypes
        elif len(run_subtypes) < 10:
            additional_needed = 10 - len(run_subtypes)
            run_subtypes_pool = [
                s
                for t in run_types
                for s in type_to_subtypes[t]
                if s not in run_subtypes
            ]
            if run_subtypes_pool:
                additional_subtypes = np.random.choice(
                    run_subtypes_pool,
                    min(additional_needed, len(run_subtypes_pool)),
                    replace=False,
                )
                run_subtypes = np.concatenate(
                    [run_subtypes, additional_subtypes]
                ).tolist()
        for t in run_types:
            if not any(s.startswith(t) for s in run_subtypes):
                available_subtypes = type_to_subtypes[t]
                run_subtypes.append(np.random.choice(available_subtypes))
                if len(run_subtypes) > 20:
                    other_subtypes = [s for s in run_subtypes if not s.startswith(t)]
                    run_subtypes = [s for s in run_subtypes if s.startswith(t)] + list(
                        np.random.choice(
                            other_subtypes,
                            20 - sum(s.startswith(t) for s in run_subtypes),
                            replace=False,
                        )
                    )
        run_data_chunk = pd.DataFrame(
            {
                "run_id": [run_id] * rows_per_run,
                "text": [
                    f"Sample text {run_idx * rows_per_run + i}"
                    for i in range(rows_per_run)
                ],
                "true_type": np.random.choice(run_types, rows_per_run),
            }
        )
        run_data_chunk["true_subtype"] = run_data_chunk["true_type"].apply(
            lambda t: np.random.choice([s for s in run_subtypes if s.startswith(t)])
        )
        run_data_chunk["pred_subtype"] = run_data_chunk["true_type"].apply(
            lambda t: np.random.choice([s for s in run_subtypes if s.startswith(t)])
        )
        run_data_chunk["correct"] = (
            run_data_chunk["true_subtype"] == run_data_chunk["pred_subtype"]
        )
        np.random.seed(43 + run_idx)
        run_data_chunk["pred_unk_subtype"] = np.random.choice(
            all_unk_subtypes, size=rows_per_run, p=[0.09] * 10 + [0.1]
        )
        detailed_data.append(run_data_chunk)
    detailed_data = pd.concat(detailed_data, ignore_index=True)
    detailed_data = pd.concat([detailed_data, test_run_data], ignore_index=True)
    unique_benchmarks = sorted(run_data["benchmark"].unique())
    return run_data, detailed_data, test_run_id, unique_benchmarks

# Create benchmark options
benchmark_options = [{"label": bm, "value": bm} for bm in generate_mock_data()[3]]