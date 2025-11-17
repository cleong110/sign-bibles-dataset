from __future__ import annotations

import itertools
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import wandb
from scipy.signal import find_peaks
from tqdm.autonotebook import tqdm

# ---------------------------------------------------------------------
# Logging configuration (works cleanly in Jupyter)
# ---------------------------------------------------------------------
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_file(df: pd.DataFrame, column_to_parse="file") -> pd.DataFrame:
    df["file_stem"] = df[column_to_parse].apply(lambda f: Path(f).stem.split(".")[0])
    df["file_stem"] = df["file_stem"].str.replace(
        "ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_", ""
    )
    df["cbt_number"] = df["file_stem"].apply(lambda f: int(f.split("-")[1]) if f.lower().startswith("cbt") else None)
    video_types = ["Intro", "Passage", "More Info"]
    df["cbt_video_type"] = df["file_stem"].apply(
        lambda f: next((v for v in video_types if v.lower() in f.lower()), None)
    )

    # Video title, e.g. "God Creates The World"
    df["title"] = df["file_stem"].apply(
        lambda f: f.split("_")[1].strip().title() if f.lower().startswith("cbt") else None
    )
    df["title"] = df["title"].astype("string")
    return df


def parse_and_edit_title(window_embeddings_df):
    window_embeddings_df["file"] = window_embeddings_df["video_stem"].str.replace(
        "ase_chronological_bible_translation_in_american_sign_language_119_introductions_and_passages_cbt", "CBT"
    )
    window_embeddings_df["file"] = window_embeddings_df["file"].str.replace("passage", "Passage")
    window_embeddings_df = parse_file(window_embeddings_df)
    window_embeddings_df["title"] = window_embeddings_df["title"].str.title()
    return window_embeddings_df


def compute_micro_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute micro-averaged precision, recall, f1 from per-video metrics DataFrame."""
    required_cols = {"true_positives", "false_positives", "false_negatives"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {required_cols}")

    total_tp = int(df["true_positives"].sum())
    total_fp = int(df["false_positives"].sum())
    total_fn = int(df["false_negatives"].sum())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"micro_precision": precision, "micro_recall": recall, "micro_f1": f1}


def match_predictions(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    window: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match predictions to ground truth within a ±window (in ms).
    Loops over ground-truth entries to ensure FN are counted correctly even if
    preds_df is empty.
    """
    # At the start of match_predictions
    n_null_preds = preds_df["Sign"].isna().sum()
    n_null_gt = gt_df["Sign"].isna().sum()

    if n_null_preds > 0:
        logger.info(f"Warning: Dropping {n_null_preds} predictions with null Sign")
    if n_null_gt > 0:
        logger.info(f"Warning: Dropping {n_null_gt} ground truth entries with null Sign")

    preds_df = preds_df[preds_df["Sign"].notna()].copy()
    gt_df = gt_df[gt_df["Sign"].notna()].copy()

    preds_out = preds_df.copy()
    preds_out["correct"] = False
    gt_out = gt_df.copy()
    gt_out["found"] = False

    # Track which predictions have been matched to avoid double counting
    matched_preds = set()
    # Loop over GT rows
    for gt_idx, gt_row in gt_df.iterrows():
        gt_sign = gt_row["Sign"]
        gt_ts = gt_row["window_midpoint_ms"]
        # find matching prediction(s) within window
        candidates = preds_out[
            (preds_out["Sign"] == gt_sign)
            & (~preds_out.index.isin(matched_preds))
            & (abs(preds_out["window_midpoint_ms"] - gt_ts) <= window)
        ]
        if not candidates.empty:
            pred_idx = candidates.index[0]  # match first candidate
            preds_out.at[pred_idx, "correct"] = True
            gt_out.at[gt_idx, "found"] = True
            matched_preds.add(pred_idx)
    # --- per-sign metrics ---
    per_sign_records = []
    all_signs = sorted(set(gt_df["Sign"]).union(preds_df.get("Sign", [])))
    for sign in all_signs:
        gt_sign = gt_out[gt_out["Sign"] == sign]
        preds_sign = preds_out[preds_out["Sign"] == sign]

        # Count TP from ground truth side (each GT can only be found once)
        tp = int(gt_sign["found"].sum())
        # FP = predictions that weren't correct
        fp = int((~preds_sign["correct"]).sum())
        # FN = ground truth that wasn't found
        fn = int((~gt_sign["found"]).sum())

        prec = tp / len(preds_sign) if len(preds_sign) > 0 else float("nan")
        rec = tp / len(gt_sign) if len(gt_sign) > 0 else float("nan")
        f1_sign = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_sign_records.append(
            {
                "Sign": sign,
                "precision": prec,
                "recall": rec,
                "f1": f1_sign,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "n_preds": len(preds_sign),
                "n_gt": len(gt_sign),
            }
        )
    per_sign_df = pd.DataFrame.from_records(per_sign_records).set_index("Sign")
    return preds_out, gt_out, per_sign_df


def aggregate_metrics(per_sign_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate overall precision/recall/F1 from a per-sign stats DataFrame."""
    tp = int(per_sign_df["true_positives"].sum())
    fp = int(per_sign_df["false_positives"].sum())
    fn = int(per_sign_df["false_negatives"].sum())
    n_preds = int(per_sign_df["n_preds"].sum())
    n_gt = int(per_sign_df["n_gt"].sum())

    precision = tp / n_preds if n_preds > 0 else float("nan")
    recall = tp / n_gt if n_gt > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "n_preds": n_preds,
        "n_gt": n_gt,
    }


def unique_column_values(dataset_to_scan: ds.Dataset, column: str) -> list:
    scanner = dataset_to_scan.scanner(columns=[column])

    # Running map from value → count
    counts = {}

    for batch in scanner.to_batches():
        vc = pc.value_counts(batch[column])  # struct<values, counts>

        # Convert struct array → Python list of (value, count)
        for struct_row in vc:
            value = struct_row["values"].as_py()
            count = struct_row["counts"].as_py()

            if value in counts:
                counts[value] += count
            else:
                counts[value] = count

    # We only need the keys for distinct values
    return list(counts.keys())


def get_window_size_values_with_all_titles(dataset, target_title_count: int = 10):
    window_size_values_with_all_titles = []
    for window_size_ms in unique_column_values(dataset, "window_size_ms"):
        ds_filter = ds.field("window_size_ms") == window_size_ms
        window_df = dataset.to_table(filter=ds_filter).to_pandas()
        window_df = parse_and_edit_title(window_df)
        title_count = len(window_df["title"].unique())
        logger.info(f"{window_size_ms}: {len(window_df)} rows, {len(window_df['title'].unique())}")
        if title_count == target_title_count:
            # logger.info(f"Keeping {window_size_ms} ms")
            window_size_values_with_all_titles.append(window_size_ms)

    # # window_embeddings_df = pd.concat(collector)
    return window_size_values_with_all_titles


def load_window_embeddings_df(dataset, window_size_ms):
    ds_filter = ds.field("window_size_ms") == window_size_ms
    window_df = dataset.to_table(filter=ds_filter).to_pandas()
    window_df = parse_and_edit_title(window_df)
    return window_df


def load_osb_ground_truth(osb_glosses_version: str = "1_4"):
    osb_glosses_dir = (
        Path(r"C:\Users\Colin Leong\projects\SignBiblesDatasetPaper\Avodah_Annotations")
        / f"osb_glosses_v{osb_glosses_version}"
        / "osb_glosses"
    )
    osb_glosses_dfs = []

    for truth_file in osb_glosses_dir.rglob("*.tsv"):
        truth_file_df = pd.read_csv(truth_file, delimiter="\t")
        # truth_file_df['video_title'] = truth_file.stem # alread included.
        truth_file_df = parse_file(truth_file_df)
        osb_glosses_dfs.append(truth_file_df)
    truth_df = pd.concat(osb_glosses_dfs)
    truth_df["window_midpoint_ms"] = truth_df["midpoint_sec"] * 1000
    truth_df = truth_df.rename(columns={"Sign": "Original Sign", "ASL Citizen Gloss": "Sign"})
    return truth_df


def load_isolated_embeddings(
    isolated_sign_embeddings_dir: Path = Path(r"C:\Users\Colin Leong\data\isolated_sign_embeddings\embeddings"),
):
    # isolated_sign_embeddings_dir =
    isolated_signs_dataset = ds.dataset(isolated_sign_embeddings_dir, format="parquet", partitioning="hive")
    ds_filter = (ds.field("dataset") == "asl-citizen") & (ds.field("embedding_model") == "asl_finetune_checkpoint_best")
    embeddings_df = isolated_signs_dataset.to_table(filter=ds_filter).to_pandas()
    return embeddings_df


def find_peaks_in_df(
    df: pd.DataFrame,
    group_col="Sign",
    x_col="window_midpoint_ms",
    y_col="score",
    prominence=0.6,
    height=None,
    width=None,  # in number of samples. TODO: set based on step size?
    distance=None,
) -> pd.DataFrame:
    """
    Find peaks in each group and return a DataFrame of peak positions.

    Returns:
    - DataFrame with columns [group_col, x_col, y_col]

    """
    peak_rows = []
    for query_id, group in df.groupby(group_col):
        group = group.sort_values(x_col)
        x = group[x_col].values
        y = group[y_col].values

        # TODO: box plots of properties to investigate
        peaks, properties = find_peaks(y, prominence=prominence, height=height, width=width, distance=distance)
        for i in peaks:
            peak_rows.append(
                {
                    group_col: query_id,
                    x_col: x[i],
                    y_col: y[i],
                }
            )
    return pd.DataFrame(peak_rows, columns=[group_col, x_col, y_col])


if __name__ == "__main__":
    window_embeddings_folder = Path(
        r"C:\Users\Colin Leong\data\DBL\signclip_embeddings\window_embeddings\ase\hive_format"
    )
    osb_glosses_version: str = "1_4"

    osb_glosses_truth_df = load_osb_ground_truth(osb_glosses_version)
    isolated_sign_embeddings_df = load_isolated_embeddings()

    dataset = ds.dataset(window_embeddings_folder, format="parquet", partitioning="hive")
    # Filter out null signs from ground truth at the start
    osb_glosses_truth_df = osb_glosses_truth_df[osb_glosses_truth_df["Sign"].notna()].copy()
    n_null_signs = len(osb_glosses_truth_df) - len(
        osb_glosses_truth_df[osb_glosses_truth_df["Sign"].notna()]
    )  # TODO: fix this Futurewarning
    if n_null_signs > 0:
        logger.info(f"Filtered out {n_null_signs} ground truth entries with null Sign")

    # --- hyperparameter grids ---
    # height_multipliers = [1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]
    # height_multipliers = list(np.arange(1.0, 1.5, 0.01))
    # height_multipliers = list(np.linspace(1.00, 1.5, 200))
    # height_multipliers = list(np.linspace(0.00, 1.5, 200))
    # height_multipliers = list(np.linspace(1.209, 1.242, 200))
    # height_multipliers = list(np.linspace(1.18, 1.28, 200))
    # height_multipliers = list(np.linspace(0.8, 1.5, 300))
    # height_multipliers = list(np.linspace(1.16, 1.28, 30))
    # height_multipliers = list(np.arange(1.21, 1.25, 0.001))
    # height_multipliers = list(np.linspace(1.21, 1.23, 100))
    # height_multipliers = list(np.random.uniform(low=0.0, high=1.5, size=25))
    # height_multipliers = list(np.random.uniform(low=0.0, high=1.40, size=20))
    # height_multipliers = list(np.random.uniform(low=1.15, high=1.25, size=20))
    height_multipliers = [1.2132332381708744]
    random.shuffle(height_multipliers)
    height_multipliers = height_multipliers[:20]

    # prominences = [0.5, 0.6, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 10.0, 100.0, 1000.0]
    # prominences = list(np.arange(0.1, 100.0, 0.1))
    # prominences = list(np.logspace(1.2, 10, 100)) # a wide range!
    # prominences = list(np.logspace(-3, 2, 100))  # smaller range
    # prominences = list(np.logspace(1.1, 1.9, 50))
    # prominences = list(np.arange(18.0, 34.0, 0.1))
    # prominences = list(np.arange(7.0, 18.0, 0.1))
    # prominences = list(np.arange(19.0, 22.0, 0.01))
    # prominences = list(np.linspace(0, 22.0, 100))
    # prominences = list(np.linspace(20.0, 30.0, 100))
    # prominences = list(np.linspace(1.0, 19.0, 100))
    # prominences = list(np.linspace(1.0, 30.0, 100))
    # prominences = list(np.linspace(1.0, 30.0, 30))
    # prominences = list(np.linspace(0.0, 50.0, 200))
    # prominences = list(np.linspace(0.0, 30.0, 200))
    # prominences = list(np.linspace(0.0, 22.0, 200))
    # prominences = list(np.logspace(-3, 2, 200)) # not much point going past 10^2, F1 hits zero around 50
    # prominences = list(np.linspace(0.0, 30.0, 100))
    # prominences = list(np.linspace(19.0, 22.0, 100))
    # prominences = list(np.random.uniform(low=0.0, high=50, size=20))
    # prominences = list(np.random.uniform(low=0.0, high=29, size=20))
    prominences = [1.6130612744069568]
    random.shuffle(prominences)
    prominences = [*prominences[:20], None]

    # widths = [None, 50, 100, 200, 500, 1000, 10000]
    # widths = [None, 1, 10, 100, 1000]
    # widths = [None, 1, 2]
    # widths = list(np.logspace(0.1, 10.0, 50))
    # widths = list(np.logspace(0.0, 1.0, 50)) # 1 to 10
    # widths = list(np.logspace(0.0, 2.0, 200))  # 1 to 100
    # widths = list(np.logspace(-2.0, 0.0, 50)) # 0.01 to 1
    # widths = list(np.linspace(0.0, 20.0, 200))
    # widths = list(np.linspace(0.0, 50.0, 200))  # hits 0 around here, not much point going higher
    # widths = list(np.linspace(0.0, 15.0, 200))
    # widths = [*list(np.linspace(0.0, 10.0, 100)), None]
    # widths = [*list(np.linspace(0.0, 7.0, 100)), None]
    # widths = list(np.linspace(0.0, 6.952, 200))
    # widths = list(np.random.uniform(low=0.0, high=50, size=20))
    # widths = list(np.random.uniform(low=0.0, high=30, size=20))
    # widths = list(np.random.uniform(low=0.0, high=14, size=20))
    # widths = list(np.random.uniform(low=0.0, high=60, size=20))
    # widths = list(np.random.uniform(low=0.0, high=60, size=20))
    widths = [4.103367281005972]
    random.shuffle(widths)
    widths = [*widths[:20], None]

    # distances = [
    #     1,
    #     # None,
    #     2,
    #     3,
    #     4,
    #     5,
    #     10,
    #     # 100,
    # ]  # Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
    # distances = list(np.logspace(0.1, 3.0, 100))
    # distances = list(np.logspace(0.1, 6.0, 200))
    # distances = list(np.logspace(0.1, 10.0, 200))
    # distances = list(np.logspace(0.1, 10.0, 200))
    # distances = list(np.logspace(0.0, 2.0, 20)) # 1 to 100
    # distances = list(np.logspace(0.0, 4.0, 200)) # 1 to 10_000
    # distances = list(np.logspace(0.0, 5.0, 200))  # 1 to 100_000
    # distances = list(np.logspace(0.0, 6.0, 1000))  # 1 to 1_000_000
    # distances = list(np.logspace(0.0, 1.0, 50)) # 1 to 10
    # distances = list(np.arange(0.0, 10.0, 1))
    # distances = list(np.arange(6, 10.0, 0.1))
    # distances = list(np.linspace(250.0, 500.0, 50))
    # distances = list(np.linspace(1.0, 100.0, 200))
    # distances = list(np.linspace(11.0, 20.0, 100))
    # distances = [i + 1 for i in range(43)]
    # distances = [i + 1 for i in range(11)]  # must be greater than 1, should be an integer
    # distances = list(np.random.uniform(low=0.0, high=1e15, size=20))
    distances = [37]
    random.shuffle(distances)
    distances = [*distances[:20], None]

    logger.info(f"{height_multipliers=}")
    logger.info(f"{prominences=}")
    logger.info(f"{distances=}")
    logger.info(f"{widths=}")

    match_window_ms = 1000
    # example_counts = [1, 5, 10, 15, 25]
    example_counts = list(range(1, 29))
    # example_counts = [25]
    random.shuffle(example_counts)
    # example_counts = example_counts[:5]

    thresholdings = ["all-gloss", "per-gloss"]
    # thresholdings = ["all-gloss"]
    random.shuffle(thresholdings)

    # all available embedding window sizes
    # available_window_sizes = sorted(window_embeddings_df["window_size_ms"].unique())
    available_window_sizes = get_window_size_values_with_all_titles(dataset, 10)
    # available_window_sizes = [
    #     s for s in available_window_sizes if s in [300, 400, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 2000,3000,8000]
    # ]
    # available_window_sizes = [s for s in available_window_sizes if s in [1000, 1100, 1400]]
    # available_window_sizes = [s for s in available_window_sizes if s in [550, 650, 750]]
    # available_window_sizes = [s for s in available_window_sizes if s in [8000, 3000, 2000]]
    # available_window_sizes = [s for s in available_window_sizes if s not in [10000, 9000, 5000]]
    # available_window_sizes = [s for s in available_window_sizes if s in [1400]]
    # available_window_sizes = [600, 600, 600, 600, 600, 600, *available_window_sizes]
    available_window_sizes = [s for s in available_window_sizes if s <= 3000]
    random.shuffle(available_window_sizes)
    logger.info(f"{available_window_sizes=}")

    runs_per_window_size = 2  # actually per window size and example count

    for embedding_window_ms in tqdm(available_window_sizes, desc="Embedding window sweep"):
        for example_count in tqdm(example_counts, desc="Example Count sweep"):
            # load fresh every run
            window_embeddings_df = load_window_embeddings_df(dataset=dataset, window_size_ms=embedding_window_ms)
            # --- generate results_df for this embedding window, with this many examples. ---
            results = []
            sample_sizes = []  # collect per-sign n_sample values
            for (cbt_number, cbt_video_type), file_gt_df in osb_glosses_truth_df.groupby(
                ["cbt_number", "cbt_video_type"]
            ):
                title = file_gt_df["title"].iloc[0]
                file_window_embeddings_df = window_embeddings_df[
                    (window_embeddings_df["cbt_number"] == cbt_number)
                    & (window_embeddings_df["cbt_video_type"] == cbt_video_type)
                    & (window_embeddings_df["window_size_ms"] == embedding_window_ms)
                ]
                if file_window_embeddings_df.empty:
                    continue

                file_window_embeddings_array = np.vstack(file_window_embeddings_df["embedding"])

                for sign in file_gt_df["Sign"].unique():
                    sign_df = isolated_sign_embeddings_df[isolated_sign_embeddings_df["GLOSS"] == sign]
                    if sign_df.empty:
                        continue
                    n_sign_df = len(sign_df)
                    n_sample = min(n_sign_df, example_count)
                    sign_df = sign_df.sample(n=n_sample, random_state=0)
                    sign_embeddings = np.vstack(sign_df["embedding"])

                    sign_scores = sign_embeddings @ file_window_embeddings_array.T
                    mean_scores = sign_scores.mean(axis=0)

                    # record sign sampling
                    sample_sizes.append(
                        {
                            "Sign": sign,
                            "cbt_number": cbt_number,
                            "cbt_video_type": cbt_video_type,
                            "example_count_requested": example_count,
                            "available_samples_count": n_sign_df,
                            "n_sample_used": n_sample,
                            "sign_df_len_after_sampling": len(sign_df),
                        }
                    )

                    for idx, score in enumerate(mean_scores):
                        results.append(
                            {
                                "cbt_number": cbt_number,
                                "cbt_video_type": cbt_video_type,
                                "Sign": sign,
                                "window_index": idx,
                                "window_midpoint_ms": file_window_embeddings_df["window_midpoint_ms"].iloc[idx],
                                "score": score,
                                "title": title,
                                "n_sign_df": n_sign_df,
                                "n_sample": n_sample,
                            }
                        )

            results_df = pd.DataFrame(results)
            if results_df.empty:
                continue
            logger.info(f"Got {len(results_df)} results")

            sample_sizes_df = pd.DataFrame(sample_sizes)

            # --- sweep over hyperparameters, including thresholding ---
            hyperparam_combos = list(
                itertools.product(height_multipliers, prominences, widths, thresholdings, distances)
            )

            random.shuffle(hyperparam_combos)
            hyperparam_combos = hyperparam_combos[:runs_per_window_size]

            for height_multiplier, prominence, width, thresholding, distance in tqdm(
                hyperparam_combos, desc=f"Threshold sweep for {embedding_window_ms}ms"
            ):
                sampling_records: list[dict[str, int | str]] = []
                logger.info(
                    "height_multiplier=%s, prominence=%s, width=%s, example_count=%s, thresholding=%s, distance=%s",
                    height_multiplier,
                    prominence,
                    width,
                    example_count,
                    thresholding,
                    distance,
                )

                logger.info(f"{width=}")
                run = wandb.init(
                    project="sign-bibles-matching",
                    config={
                        "embedding_window_ms": embedding_window_ms,
                        "height_multiplier": height_multiplier,
                        "prominence": prominence,
                        "width": width,
                        "match_window_ms": match_window_ms,
                        "example_count": example_count,
                        "thresholding": thresholding,
                        "distance": distance,
                    },
                    reinit="create_new",
                    tags=[
                        "metrics_v3",
                        f"osb_glosses_v{osb_glosses_version}",
                        "sampling_fixed_v2",
                        "pyarrow_ds_loading",
                    ],
                )

                combo_video_metrics = []
                combo_gloss_metrics = []

                for (cbt_number, cbt_video_type), group_df in results_df.groupby(["cbt_number", "cbt_video_type"]):
                    file_gt_df = osb_glosses_truth_df[
                        (osb_glosses_truth_df["cbt_number"] == cbt_number)
                        & (osb_glosses_truth_df["cbt_video_type"] == cbt_video_type)
                    ]
                    peaks_df = None
                    if thresholding == "all-gloss":
                        group_mean = group_df["score"].mean()
                        peaks_df = find_peaks_in_df(
                            group_df,
                            height=group_mean * height_multiplier,
                            prominence=prominence,
                            width=width,
                            distance=distance,
                        )

                    elif thresholding == "per-gloss":
                        per_gloss_collector = []
                        for sign, sign_results_df in group_df.groupby("Sign"):
                            sign_mean = sign_results_df["score"].mean()
                            sign_peaks = find_peaks_in_df(
                                sign_results_df,
                                height=sign_mean * height_multiplier,
                                prominence=prominence,
                                width=width,
                                distance=distance,
                            )
                            per_gloss_collector.append(sign_peaks)
                        # TODO: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
                        peaks_df = (
                            pd.concat(per_gloss_collector)
                            if per_gloss_collector
                            else pd.DataFrame(columns=group_df.columns)
                        )
                    else:
                        raise NotImplementedError(f"Thresholding {thresholding}")

                    preds_out, gt_out, per_sign_df = match_predictions(peaks_df, file_gt_df, window=match_window_ms)
                    video_metrics = aggregate_metrics(per_sign_df)
                    video_metrics.update(
                        {
                            "cbt_number": cbt_number,
                            "cbt_video_type": cbt_video_type,
                            "thresholding": thresholding,
                        }
                    )
                    combo_video_metrics.append(video_metrics)

                    gloss_df = per_sign_df.reset_index().assign(
                        cbt_number=cbt_number,
                        cbt_video_type=cbt_video_type,
                        thresholding=thresholding,
                    )
                    combo_gloss_metrics.append(gloss_df)

                video_metrics_df = pd.DataFrame(combo_video_metrics)
                gloss_metrics_df = pd.concat(combo_gloss_metrics, ignore_index=True)

                run.log(compute_micro_metrics(gloss_metrics_df))
                run.log({"video_metrics": wandb.Table(dataframe=video_metrics_df)})
                run.log({"gloss_metrics": wandb.Table(dataframe=gloss_metrics_df)})

                # --- Group by Sign and sum the count columns ---
                grouped_metrics = (
                    gloss_metrics_df.groupby("Sign")
                    .agg(
                        {
                            "true_positives": "sum",
                            "false_positives": "sum",
                            "false_negatives": "sum",
                            "n_preds": "sum",
                            "n_gt": "sum",
                        }
                    )
                    .reset_index()
                )
                # --- Compute micro-metrics per Sign ---
                sign_micro_metrics = []
                for _, row in grouped_metrics.iterrows():
                    # build a single-row DataFrame with required count columns
                    counts_df = pd.DataFrame(
                        [
                            {
                                "true_positives": row["true_positives"],
                                "false_positives": row["false_positives"],
                                "false_negatives": row["false_negatives"],
                            }
                        ]
                    )
                    metrics = compute_micro_metrics(counts_df)
                    # combine counts and computed metrics
                    metrics.update(
                        {
                            "Sign": row["Sign"],
                            "true_positives": row["true_positives"],
                            "false_positives": row["false_positives"],
                            "false_negatives": row["false_negatives"],
                            "n_preds": row["n_preds"],
                            "n_gt": row["n_gt"],
                        }
                    )
                    sign_micro_metrics.append(metrics)

                # --- Convert to DataFrame and reorder columns for readability ---
                sign_micro_df = pd.DataFrame(sign_micro_metrics)
                sign_micro_df = sign_micro_df[
                    [
                        "Sign",
                        "micro_precision",
                        "micro_recall",
                        "micro_f1",
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                        "n_preds",
                        "n_gt",
                    ]
                ]

                # --- Log to W&B ---
                run.log({"sampling_info": wandb.Table(dataframe=sample_sizes_df)})

                run.log({"grouped_sign_metrics": wandb.Table(dataframe=sign_micro_df)})

                run.finish()
