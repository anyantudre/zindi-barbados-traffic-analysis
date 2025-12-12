import pandas as pd
import numpy as np
import ast
from sklearn.feature_selection import VarianceThreshold


def preprocess_simple_features(
    simple_path: str = "/kaggle/input/simple-features-for-traffic-analysis/simple_features.csv",
    upgraded_path: str = "/kaggle/input/tourists-barbados-features/upgraded_simple_features.csv",
    output_path: str = "simple_features_merged.csv"
):
    simple = pd.read_csv(simple_path)
    upgraded = pd.read_csv(upgraded_path)

    # Merge
    merged = simple.merge(upgraded, on="video_path", how="left")

    # Drop duplicated frame-count column
    if "num_frames_sampled_y" in merged.columns:
        merged = merged.drop(columns=["num_frames_sampled_y"])

    # Fix naming to a single column name
    merged = merged.rename(columns={"num_frames_sampled_x": "num_frames_sampled"})

    merged.to_csv(output_path, index=False)

    return merged

def preprocess_yolo_features(
    upgraded_path: str = "/kaggle/input/tourists-barbados-features/upgraded_yolo_features.csv",
    simple_path: str = "/kaggle/input/tourists-barbados-features/yolo_features.csv",
    var_threshold=0.01,
    output_path: str = "yolo_features_merged.csv"
):
    upgraded = pd.read_csv(upgraded_path)
    simple = pd.read_csv(simple_path)

    # Drop useless all-NaN classes
    cols_to_drop = [
        "class_count_elephant", "class_count_bird", "class_count_tv",
        "class_count_potted plant", "class_count_dog", "class_count_umbrella",
        "class_count_chair"
    ]
    upgraded = upgraded.drop(columns=cols_to_drop, errors="ignore")

    # Replace remaining NaNs with 0
    df = upgraded.fillna(0)

    # Ensure evolution column is list
    def ensure_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                out = ast.literal_eval(x)
                return out if isinstance(out, list) else []
            except Exception:
                return []
        return []

    df["object_count_evolution"] = df["object_count_evolution"].apply(ensure_list)

    # Derived evolution features
    df["obj_evo_mean"] = df["object_count_evolution"].apply(lambda x: np.mean(x) if len(x) else 0)
    df["obj_evo_std"] = df["object_count_evolution"].apply(lambda x: np.std(x) if len(x) else 0)
    df["obj_evo_max"] = df["object_count_evolution"].apply(lambda x: np.max(x) if len(x) else 0)
    df["obj_evo_min"] = df["object_count_evolution"].apply(lambda x: np.min(x) if len(x) else 0)
    df["obj_evo_len"] = df["object_count_evolution"].apply(len)

    def compute_slope(seq):
        if len(seq) <= 1:
            return 0
        x = np.arange(len(seq))
        return np.polyfit(x, seq, 1)[0]

    df["obj_evo_slope"] = df["object_count_evolution"].apply(compute_slope)

    # Expand evolution into obj_evo_i columns
    max_len = df["object_count_evolution"].apply(len).max()
    for i in range(max_len):
        df[f"obj_evo_{i}"] = df["object_count_evolution"].apply(
            lambda x: x[i] if i < len(x) else 0
        )

    df = df.drop(columns=["object_count_evolution"])

    # VarianceThreshold filtering
    selector = VarianceThreshold(threshold=var_threshold)
    selector.fit(df.drop(columns=["video_path"]))

    keep = df.drop(columns=["video_path"]).columns[selector.get_support()]
    drop_lowvar = df.drop(columns=["video_path"]).columns[~selector.get_support()]

    df = df[["video_path"] + keep.tolist()]

    # Merge simple YOLO stats
    merged = df.merge(simple, on="video_path", how="left")

    # Remove perfectly correlated features
    X = merged.drop(columns=["video_path"])
    corr = X.corr()

    to_drop = set()
    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iloc[i, j]) == 1.0:
                to_drop.add(cols[j])  # keep cols[i], drop cols[j]

    merged = merged.drop(columns=list(to_drop))

    merged.to_csv(output_path, index=False)

    return merged, drop_lowvar, to_drop


final_df, lowvar_cols, perfect_corr_cols = preprocess_yolo_features(
    upgraded_path="/kaggle/input/tourists-barbados-features/upgraded_yolo_features.csv",
    simple_path="/kaggle/input/tourists-barbados-features/yolo_features.csv",
    var_threshold=0.01,
    output_path="yolo_features_merged_rerun.csv"
)