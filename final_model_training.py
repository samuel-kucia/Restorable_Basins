#!/usr/bin/env python3
"""Train the final XGBoost wetland model from the public input files."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xgboost as xgb
from shapely.geometry import box


# ─── FILE INPUTS ────────────────────────────────────────────────────────────────

# Place these three files in the same folder as this script:
#   StudyArea_AOI.shp (+ its .dbf, .shx, .prj, and other sidecars)
#   X_Inputs.tif
#   Y_TrainingPoints.gpkg
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

AOI_PATH = SCRIPT_DIR / "StudyArea_AOI.shp"
PREDICTOR_RASTER = SCRIPT_DIR / "X_Inputs.tif"
TRAINING_POINTS = SCRIPT_DIR / "Y_TrainingPoints.gpkg"


# ─── MODEL SETTINGS ────────────────────────────────────────────────────────────

CRS = "EPSG:32614"
BLOCK_SIZE_M = 600.0
VALIDATION_BLOCK_FRACTION = 0.20
EARLY_STOPPING_ROUNDS = 150
BOOSTING_ROUNDS = 6000
RANDOM_SEED = 42

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "gamma": 1.0,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "max_bin": 256,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "seed": RANDOM_SEED,
}


# ─── OUTPUTS ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = SCRIPT_DIR / "Outputs"
OUTPUT_MODEL = OUTPUT_DIR / "final_model.json"


def check_inputs():
    """Check that all required input files are available."""
    required_files = [AOI_PATH, PREDICTOR_RASTER, TRAINING_POINTS]
    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(f"  {path}" for path in missing_files)
        raise FileNotFoundError(f"Required input file(s) not found:\n{missing}")


def make_spatial_validation_split(points, train_bounds):
    """Hold out 20% of spatial blocks for early stopping."""
    x_min, y_min, _, _ = train_bounds
    x = points.geometry.x.to_numpy()
    y = points.geometry.y.to_numpy()

    block_x = np.floor((x - x_min) / BLOCK_SIZE_M).astype(int)
    block_y = np.floor((y - y_min) / BLOCK_SIZE_M).astype(int)
    block_coordinates = np.column_stack((block_x, block_y))
    _, block_ids = np.unique(
        block_coordinates,
        axis=0,
        return_inverse=True,
    )

    unique_blocks = np.unique(block_ids)
    if unique_blocks.size < 2:
        raise ValueError(
            "Fewer than two spatial blocks were found. Reduce BLOCK_SIZE_M "
            "or check the training points."
        )

    rng = np.random.default_rng(13)
    rng.shuffle(unique_blocks)
    n_validation_blocks = max(
        1,
        int(np.ceil(VALIDATION_BLOCK_FRACTION * unique_blocks.size)),
    )
    validation_blocks = unique_blocks[:n_validation_blocks]

    validation_mask = np.isin(block_ids, validation_blocks)
    training_mask = ~validation_mask
    return training_mask, validation_mask


def main():
    check_inputs()

    # Define the western 70% of the study area as the training region.
    aoi = gpd.read_file(AOI_PATH)
    if aoi.crs is None:
        raise ValueError(f"The AOI has no defined CRS: {AOI_PATH}")

    aoi = aoi.to_crs(CRS)
    x_min, y_min, x_max, y_max = aoi.total_bounds
    split_x = x_min + 0.70 * (x_max - x_min)
    training_area = box(x_min, y_min, split_x, y_max)
    training_bounds = (x_min, y_min, split_x, y_max)

    # Read the pre-generated wetland (1) and non-wetland (0) training points.
    points = gpd.read_file(TRAINING_POINTS)
    if points.crs is None:
        raise ValueError(
            f"The training points have no defined CRS: {TRAINING_POINTS}"
        )
    if "class" not in points.columns:
        raise ValueError("Y_TrainingPoints.gpkg must contain a 'class' field.")

    points = points[
        points.geometry.notna() & ~points.geometry.is_empty
    ].copy()
    points = points.to_crs(CRS)
    points["class"] = points["class"].astype(int)

    if set(points["class"].unique()) != {0, 1}:
        raise ValueError("Training points must include classes 0 and 1.")

    outside_training_area = ~points.geometry.within(training_area)
    if outside_training_area.any():
        raise ValueError(
            f"{outside_training_area.sum():,} training point(s) fall outside "
            "the western 70% training region."
        )

    # Extract predictor values at each training point.
    coordinates = [(point.x, point.y) for point in points.geometry]

    with rasterio.open(PREDICTOR_RASTER) as raster:
        if raster.crs is None:
            raise ValueError(
                f"The predictor raster has no defined CRS: {PREDICTOR_RASTER}"
            )
        if raster.crs.to_string() != CRS:
            raise ValueError(
                f"Expected raster CRS {CRS}; found {raster.crs.to_string()}."
            )

        samples = np.ma.vstack(
            list(raster.sample(coordinates, masked=True))
        ).astype(np.float32)
        band_count = raster.count
        feature_names = [
            description or f"f{index}"
            for index, description in enumerate(raster.descriptions)
        ]

    features = samples.filled(np.nan)
    valid_rows = (
        ~np.ma.getmaskarray(samples).any(axis=1)
        & np.isfinite(features).all(axis=1)
    )

    points = points.loc[valid_rows].reset_index(drop=True)
    features = features[valid_rows]
    labels = points["class"].to_numpy(dtype=np.int32)

    if features.shape[0] < 200:
        raise ValueError(
            f"Only {features.shape[0]:,} valid training samples remain "
            "after removing raster NoData pixels."
        )

    if len(feature_names) != band_count:
        feature_names = [f"f{index}" for index in range(band_count)]

    training_mask, validation_mask = make_spatial_validation_split(
        points,
        training_bounds,
    )

    x_train = features[training_mask]
    y_train = labels[training_mask]
    x_validation = features[validation_mask]
    y_validation = labels[validation_mask]

    if np.unique(y_train).size < 2 or np.unique(y_validation).size < 2:
        raise ValueError(
            "Both classes must be present in the training and validation "
            "sets. Check the points and spatial block split."
        )

    positive_count = np.count_nonzero(y_train == 1)
    negative_count = np.count_nonzero(y_train == 0)
    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = negative_count / positive_count

    dtrain = xgb.DMatrix(
        x_train,
        label=y_train,
        feature_names=feature_names,
    )
    dvalidation = xgb.DMatrix(
        x_validation,
        label=y_validation,
        feature_names=feature_names,
    )

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=BOOSTING_ROUNDS,
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    # Save the trained model.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_MODEL))

    print(f"Training points used: {len(points):,}")
    print(f"Predictor bands used: {band_count}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Model saved to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
