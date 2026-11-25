import numpy as np


def create_engineered_features(data):
    df = data.copy()

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    employed_days = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYED_YEARS"] = -employed_days / 365.25

    df["CREDIT_TO_INCOME"] = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )
    df["ANNUITY_TO_INCOME"] = (
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )
    df["CREDIT_TO_ANNUITY"] = (
        df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    )
    df["GOODS_TO_CREDIT"] = (
        df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)
    )
    df["INCOME_PER_FAMILY_MEMBER"] = (
        df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    )

    external_scores = df[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ]
    df["EXT_SOURCE_MEAN"] = external_scores.mean(axis=1)
    df["EXT_SOURCE_MAX"] = external_scores.max(axis=1)

    return df