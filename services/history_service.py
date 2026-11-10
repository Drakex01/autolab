import pandas as pd
from typing import List, Dict, Any

def format_runs_for_comparison(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flattens a list of experiment runs into a comparative pandas DataFrame."""
    if not runs:
        return pd.DataFrame()

    rows = []
    for r in runs:
        row = {
            "Run ID": r["id"],
            "Dataset": r["dataset_name"],
            "Type": r["experiment_type"],
            "Model": r["config"].get("model_type", r["config"].get("model_name", "N/A")),
            "Timestamp": r["created_at"][:19],
        }
        # Add metrics
        for m_name, m_val in r["metrics"].items():
            row[f"Metric: {m_name}"] = m_val

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
