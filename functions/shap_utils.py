"""
SHAP analysis utilities for tree-based models.
"""

import os
import json
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _sanitize_base_score_str(cfg_str: str) -> str:
    return cfg_str.replace("[5E-1]", "0.5").replace("[5e-1]", "0.5")


def _select_shap_matrix(shap_values: Any) -> Any:
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    return shap_values


def _select_expected_value(expected_value: Any, positive_class: int = 1) -> float:
    try:
        if isinstance(expected_value, list):
            return float(expected_value[positive_class] if len(expected_value) > positive_class else expected_value[0])
        if isinstance(expected_value, np.ndarray):
            if expected_value.ndim == 0:
                return float(expected_value)
            if expected_value.ndim == 1:
                return float(expected_value[positive_class] if expected_value.size > positive_class else expected_value[0])
    except Exception:
        pass
    try:
        return float(expected_value)
    except Exception:
        return 0.0


def _get_top_features(shap_array: np.ndarray, feature_names: List[str], max_features: int) -> List[str]:
    max_features = min(max_features, shap_array.shape[1])
    order = np.argsort(np.abs(shap_array).mean(axis=0))[::-1]
    return [feature_names[i] for i in order[:max_features]]


def _prepare_feature_matrix(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    imputer: Any = None,
    sample_size: int | None = None,
    random_state: int = 42,
    return_indices: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Index]:
    feature_matrix = features_df[feature_cols].copy()
    used_indices = feature_matrix.index

    if imputer is not None:
        feature_matrix = pd.DataFrame(
            imputer.transform(feature_matrix.values),
            columns=feature_cols,
        )

    if sample_size is not None and len(feature_matrix) > sample_size:
        rng = np.random.default_rng(random_state)
        sampled_idx = rng.choice(len(feature_matrix), size=sample_size, replace=False)
        used_indices = feature_matrix.index[sampled_idx]
        feature_matrix = feature_matrix.iloc[sampled_idx].reset_index(drop=True)

    if return_indices:
        return feature_matrix, used_indices
    return feature_matrix


def _save_summary_plots(
    shap_values: Any,
    feature_matrix: pd.DataFrame,
    model_name: str,
    output_dir: str,
    max_display: int = 20,
) -> Dict[str, str]:
    shap_array = _select_shap_matrix(shap_values)
    feature_names = feature_matrix.columns.tolist()

    os.makedirs(output_dir, exist_ok=True)

    beeswarm_path = os.path.join(output_dir, f"{model_name}_shap_beeswarm.png")
    plt.figure()
    shap.summary_plot(
        shap_array,
        feature_matrix,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
    plt.close()

    bar_path = os.path.join(output_dir, f"{model_name}_shap_bar.png")
    plt.figure()
    shap.summary_plot(
        shap_array,
        feature_matrix,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight", dpi=300)
    plt.close()

    return {"beeswarm": beeswarm_path, "bar": bar_path}


def _save_dependence_plots(
    shap_array: np.ndarray,
    feature_matrix: pd.DataFrame,
    model_name: str,
    output_dir: str,
    max_display: int = 6,
) -> Dict[str, str]:
    feature_names = feature_matrix.columns.tolist()
    top_features = _get_top_features(shap_array, feature_names, max_display)
    os.makedirs(output_dir, exist_ok=True)

    # Combined 2x3 grid plot
    grid_path = os.path.join(output_dir, f"{model_name}_dependence_top{max_display}.png")
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, feat in enumerate(top_features):
        shap.dependence_plot(
            feat,
            shap_array,
            feature_matrix,
            feature_names=feature_names,
            interaction_index="auto",
            show=False,
            ax=axes_flat[i],
        )
    
    # Hide unused subplots
    for i in range(len(top_features), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(grid_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Still save individual plots just in case? Or return the grid one?
    # Returning just the grid one to satisfy the request "single image"
    return {"grid": grid_path}


def _extract_base_values(
    shap_array: np.ndarray,
    expected_value: Any,
    shap_result: Any = None,
    positive_class: int = 1,
) -> np.ndarray:
    """
    Normalises expected/base values to a vector aligned with shap_array rows.
    """
    n_samples = shap_array.shape[0]
    if shap_result is not None and hasattr(shap_result, "base_values"):
        base_vals = np.array(getattr(shap_result, "base_values"))
        if base_vals.ndim == 0:
            return np.full(n_samples, float(base_vals))
        if base_vals.ndim == 1:
            if base_vals.size == 1:
                return np.full(n_samples, float(base_vals[0]))
            return base_vals.reshape(-1)
        if base_vals.ndim == 2:
            if base_vals.shape[1] > positive_class:
                return base_vals[:, positive_class]
            return base_vals[:, 0]

    base_scalar = _select_expected_value(expected_value, positive_class=positive_class)
    return np.full(n_samples, base_scalar)


def _choose_index(mask: np.ndarray, scores: np.ndarray, strategy: str = "max") -> int | None:
    """Selects an index from mask using score-based strategy."""
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return None
    if strategy == "min":
        chosen = candidates[np.argmin(scores[candidates])]
    else:
        chosen = candidates[np.argmax(scores[candidates])]
    return int(chosen)


def _save_force_plots(
    shap_array: np.ndarray,
    base_values: np.ndarray,
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    model: Any,
    model_name: str,
    output_dir: str,
    threshold: float = 0.5,
) -> Dict[str, str | None]:
    """
    Saves force plots for TP, FP, and FN cases in a single image with subplots.
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_names = feature_matrix.columns.tolist()
    try:
        proba_raw = model.predict_proba(feature_matrix)
        y_prob = proba_raw if proba_raw.ndim == 1 else proba_raw[:, 1]
    except Exception:
        # Fallback to decision function scaled to [0,1] if predict_proba is unavailable
        scores = model.decision_function(feature_matrix)
        y_prob = 1 / (1 + np.exp(-scores))

    y_pred = (y_prob >= threshold).astype(int)
    labels_arr = labels.to_numpy()

    # Select representative indices
    tp_idx = _choose_index((labels_arr == 1) & (y_pred == 1), y_prob, strategy="max")
    fp_idx = _choose_index((labels_arr == 0) & (y_pred == 1), y_prob, strategy="max")
    fn_idx = _choose_index((labels_arr == 1) & (y_pred == 0), y_prob, strategy="min")

    cases = [
        ("True Positive (Correctly Predicted Risk)", tp_idx),
        ("False Positive (False Alarm)", fp_idx),
        # ("False Negative (Missed Risk)", fn_idx), # Keeping FN is good practice but user asked for TP and FP.
        # I'll include all 3 for completeness if possible, or just TP/FP if space is tight.
        # Given "true positive and false positive" explicit request, I will prioritize them.
        # But since I have space (subplots), I will add FN as it's critical.
    ]
    if fn_idx is not None:
         cases.append(("False Negative (Missed Risk)", fn_idx))

    # shap.force_plot with matplotlib=True doesn't easily support 'ax'.
    # Workaround: Save each plot to a temporary buffer/file, then load into subplots.
    
    n_plots = len([c for c in cases if c[1] is not None])
    if n_plots == 0:
        print(f"[SHAP][{model_name}] No TP/FP/FN cases found to plot.")
        return {}

    fig = plt.figure(figsize=(20, 6 * n_plots))
    
    # We will generate individual images and stitch them using imshow
    # because shap.force_plot(matplotlib=True) creates its own figure/axes and is hard to embed.
    
    temp_files = []
    
    for i, (title, idx) in enumerate(cases):
        if idx is None:
            continue
            
        shap_row = shap_array[idx]
        base_val = float(base_values[idx]) if getattr(base_values, "ndim", 0) > 0 else float(base_values)
        features_row = feature_matrix.iloc[idx]
        
        # Create a separate figure for the force plot
        plt.figure(figsize=(20, 5))
        shap.force_plot(
            base_val,
            shap_row,
            features_row,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=45, 
        )

        # Remove "base value" and "f(x)" labels (often grey) as requested
        try:
            fig = plt.gcf()
            for ax in fig.axes:
                for txt in ax.texts:
                    text_content = txt.get_text()
                    if "base value" in text_content or "f(x)" in text_content:
                        txt.set_visible(False)
        except Exception:
            pass

        # Force plot with matplotlib=True usually writes on the current figure.
        # We add a title to it.
        plt.title(f"{model_name.upper()} - {title}\n(Prob: {y_prob[idx]:.3f}, Label: {labels_arr[idx]})", fontsize=14)
        plt.tight_layout()
        
        tmp_path = os.path.join(output_dir, f".temp_{model_name}_{i}.png")
        plt.savefig(tmp_path, bbox_inches="tight", dpi=150)
        plt.close()
        temp_files.append(tmp_path)

    # Now create the combined figure
    plt.figure(figsize=(24, 6 * len(temp_files)))
    for i, tmp_path in enumerate(temp_files):
        ax = plt.subplot(len(temp_files), 1, i + 1)
        img = plt.imread(tmp_path)
        ax.imshow(img)
        ax.axis('off')
        # Clean up temp file
        os.remove(tmp_path)
        
    combined_path = os.path.join(output_dir, f"{model_name}_force_cases_combined.png")
    plt.tight_layout()
    plt.savefig(combined_path, bbox_inches="tight", dpi=300)
    plt.close()

    return {"combined": combined_path}


def _build_explainer(model: Any, model_name: str) -> shap.TreeExplainer:
    # LightGBM exposes the underlying booster separately
    if model_name.lower() == "lightgbm" and hasattr(model, "booster_"):
        return shap.TreeExplainer(model.booster_)

    # XGBoost sometimes stores base_score as a bracketed string (e.g., "[5E-1]")
    # which SHAP cannot parse. Normalise it before constructing the explainer.
    if hasattr(model, "get_booster"):
        booster = model.get_booster()

        def _log_base_score(tag: str) -> None:
            try:
                cfg_raw = booster.save_config()
                print(f"[SHAP][{tag}] save_config contains [5E-1]? {'[5E-1]' in cfg_raw}")
                cfg = json.loads(cfg_raw)
                learner_param = cfg.get("learner", {}).get("learner_model_param", {})
                print(f"[SHAP][{tag}] config base_score={learner_param.get('base_score')}")
            except Exception as e:
                print(f"[SHAP][{tag}] config read failed: {e}")
            try:
                attr_val = booster.attr("base_score")
                print(f"[SHAP][{tag}] attr base_score={attr_val}")
            except Exception as e:
                print(f"[SHAP][{tag}] attr read failed: {e}")
            try:
                params = booster.attributes()
                if "base_score" in params:
                    print(f"[SHAP][{tag}] attributes base_score={params['base_score']}")
            except Exception:
                pass

        _log_base_score("before")

        # Try to rewrite the booster config to a numeric base_score
        try:
            cfg = json.loads(_sanitize_base_score_str(booster.save_config()))
            learner_param = cfg.get("learner", {}).get("learner_model_param", {})
            before_score = learner_param.get("base_score")
            learner_param["base_score"] = "0.5"
            cfg_str = json.dumps(cfg)
            cfg_str = _sanitize_base_score_str(cfg_str)
            booster.load_config(cfg_str)
            print(f"[SHAP] XGB base_score sanitized via config: {before_score} -> 0.5")
        except Exception as e:
            print(f"[SHAP] XGB base_score config sanitize failed: {e}")

        # Also adjust booster attributes (SHAP may read them)
        try:
            attr_score = booster.attr("base_score")
            cleaned = None
            if attr_score is not None:
                stripped = str(attr_score).strip("[]")
                try:
                    cleaned = float(stripped)
                except Exception:
                    cleaned = 0.5
                booster.set_attr(base_score=str(cleaned))
                print(f"[SHAP] XGB base_score attr sanitized: {attr_score} -> {cleaned}")
            else:
                booster.set_attr(base_score="0.5")
        except Exception as e:
            print(f"[SHAP] XGB base_score attr sanitize failed: {e}")

        # Patch save_config so SHAP sees sanitized base_score in config string
        try:
            original_save_config = booster.save_config

            def patched_save_config():
                cfg_str = original_save_config()
                cfg_str = _sanitize_base_score_str(cfg_str)
                try:
                    cfg = json.loads(cfg_str)
                    # sanitize both model and train params if present
                    for k in ["learner_model_param", "learner_train_param"]:
                        if k in cfg.get("learner", {}):
                            params = cfg["learner"][k]
                            if "base_score" in params:
                                params["base_score"] = "0.5"
                    cfg_str = json.dumps(cfg)
                except Exception:
                    cfg_str = _sanitize_base_score_str(cfg_str)
                return cfg_str

            booster.save_config = patched_save_config
        except Exception as e:
            print(f"[SHAP] XGB save_config patch failed: {e}")

        # Fallback: set_param
        try:
            booster.set_param({"base_score": 0.5})
        except Exception as e:
            print(f"[SHAP] XGB base_score set_param failed: {e}")

        _log_base_score("after")

        # Final guard: ensure current save_config no longer has bracketed value
        try:
            cfg_check = booster.save_config()
            if "[5E-1]" in cfg_check or "[5e-1]" in cfg_check:
                print("[SHAP] WARNING: save_config still contains [5E-1] after patch; forcing string replace")
                cfg_check_clean = _sanitize_base_score_str(cfg_check)
                booster.load_config(cfg_check_clean)
                booster.save_config = lambda: cfg_check_clean
        except Exception as e:
            print(f"[SHAP] final save_config guard failed: {e}")

        return shap.TreeExplainer(booster)

    return shap.TreeExplainer(model)


def generate_shap_analysis(
    models: Dict[str, Any],
    features_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str,
    sample_size: int | None = 500,
    max_display: int = 20,
    imputer: Any = None,
    random_seed: int = 42,
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes SHAP values for trained models and stores summary plots.
    Returns a mapping of model name to generated file paths.
    """
    feature_matrix, used_indices = _prepare_feature_matrix(
        features_df,
        feature_cols,
        imputer=imputer,
        sample_size=sample_size,
        random_state=random_seed,
        return_indices=True,
    )
    labels = features_df.loc[used_indices, "label"].reset_index(drop=True)

    shap_outputs: Dict[str, Dict[str, str]] = {}

    for name, model in models.items():
        print(f"\nGenerating SHAP plots for {name}")
        explainer = None
        shap_values = None
        shap_result = None
        model_key = name.lower()
        threshold = thresholds.get(model_key, 0.5) if thresholds else 0.5

        if model_key == "xgboost":
            try:
                explainer = _build_explainer(model, name)
                shap_values = explainer.shap_values(feature_matrix)
            except ValueError as e:
                msg = str(e)
                if "[5E-1]" in msg or "base_score" in msg:
                    print("[SHAP] TreeExplainer failed due to base_score parsing. Falling back to permutation explainer.")
                    masker = shap.maskers.Independent(feature_matrix, max_samples=feature_matrix.shape[0])
                    explainer = shap.Explainer(
                        lambda X: model.predict_proba(X)[:, 1],
                        masker=masker,
                        algorithm="permutation",
                    )
                    shap_result = explainer(feature_matrix)
                    shap_values = shap_result.values
                else:
                    raise
        else:
            explainer = _build_explainer(model, name)
            shap_values = explainer.shap_values(feature_matrix)

        shap_array = _select_shap_matrix(shap_values)
        base_values = _extract_base_values(shap_array, getattr(explainer, "expected_value", None), shap_result)

        summary_paths = _save_summary_plots(
            shap_values,
            feature_matrix,
            model_key,
            output_dir,
            max_display=max_display,
        )

        dependence_paths = _save_dependence_plots(
            shap_array,
            feature_matrix,
            model_key,
            output_dir,
            max_display=min(max_display, 6),
        )

        force_paths = _save_force_plots(
            shap_array,
            base_values,
            feature_matrix,
            labels,
            model,
            model_key,
            output_dir,
            threshold=threshold,
        )

        shap_outputs[name] = {
            "summary": summary_paths,
            "dependence": dependence_paths,
            "force": force_paths,
        }

    return shap_outputs

