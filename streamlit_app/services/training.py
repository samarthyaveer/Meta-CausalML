from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from ..constants import LEARNER_OPTIONS
from ..models import LearnerArtifacts, RunConfig, TrainingResult, ValidationReport
from .causalml_adapter import build_supported_learner, learner_label
from .evaluation import calculate_metrics
from .explanations import build_feature_importance, build_segment_summary, build_top_users
from .preprocessing import prepare_data


def train_and_compare(
    df: pd.DataFrame,
    validation_report: ValidationReport,
    config: RunConfig,
    progress_callback=None,
) -> TrainingResult:
    prepared_data = prepare_data(df=df, config=config)

    learner_outputs: dict[str, LearnerArtifacts] = {}
    predictions_full = prepared_data.raw_full_df.copy()
    predictions_test = prepared_data.raw_test_df.copy()
    comparison_rows: list[dict[str, object]] = []

    total_steps = max(1, len(config.learner_names))
    for index, learner_name in enumerate(config.learner_names, start=1):
        if progress_callback:
            progress_callback(
                index - 1,
                total_steps,
                f"Training {LEARNER_OPTIONS[learner_name]}...",
            )

        estimator = build_supported_learner(
            learner_name=learner_name,
            base_model_name=config.base_model_name,
            random_state=config.random_state,
        )

        fit_kwargs = {}
        if config.propensity_col:
            fit_kwargs["p"] = prepared_data.propensity_full.loc[prepared_data.raw_train_df["row_id"]].to_numpy()

        started = perf_counter()
        estimator.fit(
            prepared_data.X_train.to_numpy(),
            prepared_data.treatment_train.to_numpy(),
            prepared_data.y_train.to_numpy(),
            **fit_kwargs,
        )
        fit_seconds = perf_counter() - started

        predict_kwargs_full = {}
        predict_kwargs_test = {}
        if config.propensity_col:
            predict_kwargs_full["p"] = prepared_data.propensity_full.to_numpy()
            predict_kwargs_test["p"] = prepared_data.propensity_test.to_numpy()

        full_cate = _flatten_prediction(
            estimator.predict(prepared_data.X_full.to_numpy(), **predict_kwargs_full)
        )
        test_cate = _flatten_prediction(
            estimator.predict(prepared_data.X_test.to_numpy(), **predict_kwargs_test)
        )

        predictions_full[f"{learner_name}_cate"] = full_cate
        predictions_test[f"{learner_name}_cate"] = test_cate

        evaluation_frame = prepared_data.raw_test_df.copy()
        evaluation_frame["outcome_numeric"] = prepared_data.y_test.to_numpy()
        evaluation_frame["treatment_binary"] = prepared_data.treatment_test.to_numpy()
        if prepared_data.true_cate_test is not None:
            evaluation_frame["true_cate_numeric"] = prepared_data.true_cate_test.to_numpy()

        true_cate_col = "true_cate_numeric" if prepared_data.true_cate_test is not None else None
        metrics, gain_curve = calculate_metrics(
            df=evaluation_frame,
            predicted_cate=pd.Series(test_cate),
            true_cate_col=true_cate_col,
        )

        feature_importance = build_feature_importance(
            prepared_data=prepared_data,
            predicted_cate=pd.Series(full_cate),
            random_state=config.random_state,
        )
        segment_summary = build_segment_summary(
            df=evaluation_frame,
            predicted_cate=pd.Series(test_cate),
        )
        top_users = build_top_users(
            df=evaluation_frame,
            predicted_cate=pd.Series(test_cate),
            top_n=25,
        )

        metrics["fit_seconds"] = fit_seconds
        metrics["predicted_ate_full"] = float(np.mean(full_cate))
        metrics["predicted_ate_test"] = float(np.mean(test_cate))

        learner_outputs[learner_name] = LearnerArtifacts(
            learner_name=learner_name,
            learner_label=learner_label(learner_name),
            estimator=estimator,
            fit_seconds=fit_seconds,
            full_cate=pd.Series(full_cate),
            test_cate=pd.Series(test_cate),
            metrics=metrics,
            gain_curve=gain_curve,
            feature_importance=feature_importance,
            segment_summary=segment_summary,
            top_users=top_users,
        )

        comparison_rows.append(
            {
                "learner": learner_name,
                "learner_label": learner_label(learner_name),
                "average_predicted_cate": metrics["average_predicted_cate"],
                "final_gain": metrics["final_gain"],
                "final_uplift": metrics["final_uplift"],
                "auuc_like": metrics["auuc_like"],
                "predicted_ate_test": metrics["predicted_ate_test"],
                "fit_seconds": fit_seconds,
                "cate_rmse": metrics.get("cate_rmse", np.nan),
                "normalized_auuc": metrics.get("normalized_auuc", np.nan),
            }
        )

        if progress_callback:
            progress_callback(
                index,
                total_steps,
                f"Finished {LEARNER_OPTIONS[learner_name]}.",
            )

    comparison_table = pd.DataFrame(comparison_rows).sort_values(
        by="final_gain", ascending=False
    )

    export_metadata = {
        "base_model_name": config.base_model_name,
        "learner_names": config.learner_names,
        "row_count": len(df),
        "feature_count": len(config.feature_cols),
        "test_size": config.test_size,
        "random_state": config.random_state,
    }

    return TrainingResult(
        config=config,
        prepared_data=prepared_data,
        validation_report=validation_report,
        learner_outputs=learner_outputs,
        comparison_table=comparison_table.reset_index(drop=True),
        predictions_full=predictions_full.reset_index(drop=True),
        predictions_test=predictions_test.reset_index(drop=True),
        export_metadata=export_metadata,
    )


def _flatten_prediction(prediction) -> np.ndarray:
    array = np.asarray(prediction)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array.ravel()
