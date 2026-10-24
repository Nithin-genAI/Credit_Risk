"""Evidence-based descriptions for saved-model features."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


ENGINEERED_FEATURES: dict[str, dict[str, Any]] = {
    "AGE_YEARS": {
        "description": "Age in years, derived from DAYS_BIRTH.",
        "units": "years",
        "basis": "training_feature_formula",
    },
    "EMPLOYED_YEARS": {
        "description": "Employment duration in years, derived from DAYS_EMPLOYED.",
        "units": "years",
        "basis": "training_feature_formula",
    },
    "CREDIT_TO_INCOME": {
        "description": "AMT_CREDIT divided by AMT_INCOME_TOTAL.",
        "units": "ratio",
        "basis": "training_feature_formula",
    },
    "ANNUITY_TO_INCOME": {
        "description": "AMT_ANNUITY divided by AMT_INCOME_TOTAL.",
        "units": "ratio",
        "basis": "training_feature_formula",
    },
    "CREDIT_TO_ANNUITY": {
        "description": "AMT_CREDIT divided by AMT_ANNUITY.",
        "units": "ratio",
        "basis": "training_feature_formula",
    },
    "GOODS_TO_CREDIT": {
        "description": "AMT_GOODS_PRICE divided by AMT_CREDIT.",
        "units": "ratio",
        "basis": "training_feature_formula",
    },
    "INCOME_PER_FAMILY_MEMBER": {
        "description": "AMT_INCOME_TOTAL divided by CNT_FAM_MEMBERS.",
        "units": "dataset amount units per family member",
        "basis": "training_feature_formula",
    },
    "INCOME_PER_CHILD": {
        "description": "AMT_INCOME_TOTAL divided by CNT_CHILDREN plus one.",
        "units": "dataset amount units per child-adjusted denominator",
        "basis": "training_feature_formula",
    },
    "EXT_SOURCE_MEAN": {
        "description": "Mean of EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3.",
        "units": "source score scale",
        "basis": "training_feature_formula",
    },
    "EXT_SOURCE_MAX": {
        "description": "Maximum of EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3.",
        "units": "source score scale",
        "basis": "training_feature_formula",
    },
}

RAW_FEATURES: dict[str, dict[str, Any]] = {
    "AMT_INCOME_TOTAL": {
        "description": "Applicant income amount recorded in the source dataset.",
        "units": "dataset amount units; currency is not specified",
    },
    "AMT_CREDIT": {
        "description": "Credit amount recorded for the application.",
        "units": "dataset amount units; currency is not specified",
    },
    "AMT_ANNUITY": {
        "description": "Annuity amount recorded for the credit application.",
        "units": "dataset amount units; payment period is not specified",
    },
    "AMT_GOODS_PRICE": {
        "description": "Goods-price amount recorded for the application.",
        "units": "dataset amount units; currency is not specified",
    },
    "CNT_CHILDREN": {
        "description": "Child count recorded in the application dataset.",
        "units": "count",
    },
    "CNT_FAM_MEMBERS": {
        "description": "Family-member count recorded in the application dataset.",
        "units": "count",
    },
    "DAYS_BIRTH": {
        "description": "Age-related day offset used by the training pipeline; "
        "AGE_YEARS is calculated as -DAYS_BIRTH / 365.25.",
        "units": "days",
    },
    "DAYS_EMPLOYED": {
        "description": "Employment-related day offset used by the training "
        "pipeline; the known 365243 sentinel is treated as missing.",
        "units": "days",
    },
    "EXT_SOURCE_1": {
        "description": "External-source score field 1. Its source-specific "
        "meaning is not documented in this project.",
        "units": "source-defined scale",
    },
    "EXT_SOURCE_2": {
        "description": "External-source score field 2. Its source-specific "
        "meaning is not documented in this project.",
        "units": "source-defined scale",
    },
    "EXT_SOURCE_3": {
        "description": "External-source score field 3. Its source-specific "
        "meaning is not documented in this project.",
        "units": "source-defined scale",
    },
}


def describe_model_feature(
    transformed_name: str,
    categorical_features: Iterable[str],
) -> dict[str, Any]:
    """Describe one transformed preprocessor feature without guessing unknowns."""
    name = str(transformed_name)
    raw_name = name
    category: str | None = None

    if name.startswith("num__"):
        raw_name = name.removeprefix("num__")
    elif name.startswith("cat__"):
        encoded_name = name.removeprefix("cat__")
        for candidate in sorted(categorical_features, key=len, reverse=True):
            prefix = f"{candidate}_"
            if encoded_name.startswith(prefix):
                raw_name = candidate
                category = encoded_name[len(prefix) :]
                break
        else:
            raw_name = encoded_name

    if category is not None:
        return {
            "description": (
                f"Model indicator for category {category!r} in source field "
                f"{raw_name}. The project does not document additional business "
                "meaning for this category."
            ),
            "units": "categorical indicator",
            "basis": "saved_preprocessor_category",
        }

    description = ENGINEERED_FEATURES.get(raw_name) or RAW_FEATURES.get(raw_name)
    if description is None:
        return {
            "description": (
                f"Source model field {raw_name!r}. A verified business definition "
                "is not included in this project's feature metadata."
            ),
            "units": None,
            "basis": "feature_name_only",
        }

    return {
        **description,
        "raw_feature": raw_name,
        "basis": description.get(
            "basis",
            "dataset_field_name_and_project_notes",
        ),
    }
