from backend.services.feature_descriptions import describe_model_feature


def test_engineered_feature_description_matches_training_formula():
    description = describe_model_feature(
        "num__CREDIT_TO_INCOME",
        categorical_features=set(),
    )

    assert description["description"] == (
        "AMT_CREDIT divided by AMT_INCOME_TOTAL."
    )
    assert description["basis"] == "training_feature_formula"
    assert description["units"] == "ratio"


def test_one_hot_feature_description_preserves_raw_field_and_category():
    description = describe_model_feature(
        "cat__NAME_EDUCATION_TYPE_Higher education",
        {"NAME_EDUCATION_TYPE"},
    )

    assert "Higher education" in description["description"]
    assert "NAME_EDUCATION_TYPE" in description["description"]
    assert description["basis"] == "saved_preprocessor_category"


def test_unknown_raw_feature_does_not_invent_business_definition():
    description = describe_model_feature(
        "num__UNDOCUMENTED_FEATURE",
        categorical_features=set(),
    )

    assert "verified business definition is not included" in description["description"]
    assert description["units"] is None
    assert description["basis"] == "feature_name_only"
