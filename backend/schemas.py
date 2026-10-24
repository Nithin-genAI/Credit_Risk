from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApplicantInput(BaseModel):
    """Single-applicant request; extra recognized model fields are permitted."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    application_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=200)
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    AMT_CREDIT: float = Field(..., gt=0)
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    CNT_CHILDREN: Optional[int] = 0
    CNT_FAM_MEMBERS: Optional[float] = None
    DAYS_BIRTH: float = Field(..., lt=0)
    DAYS_EMPLOYED: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    NAME_CONTRACT_TYPE: Optional[str] = None

    @field_validator("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
    @classmethod
    def validate_external_score(
        cls,
        value: Optional[float],
    ) -> Optional[float]:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("External credit scores must be between 0 and 1.")
        return value


class ExplanationRequest(BaseModel):
    application_id: str = Field(..., min_length=1, max_length=128)
    applicant: dict[str, Any]
    generate_narrative: bool = False
    provider_consent: bool = False


class RegulatoryQuestion(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    audience: str = Field(default="applicant", pattern="^(applicant|loan_officer)$")
    provider_consent: bool = False


class ApplicantChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=1500)


class ApplicantChatRequest(BaseModel):
    applicant: dict[str, Any]
    question: str = Field(..., min_length=3, max_length=1000)
    audience: Literal["applicant", "loan_officer"] = "applicant"
    history: list[ApplicantChatMessage] = Field(default_factory=list, max_length=8)
    provider_consent: bool = False


class BatchValidationResponse(BaseModel):
    valid: bool
    file_type: str
    row_count: int
    recognized_model_columns: list[str]
    recognized_model_column_count: int
    expected_model_column_count: int
    missing_required_columns: list[str]
    missing_optional_model_columns: list[str]
    duplicate_columns: list[str]
    unsupported_columns: list[str]
    target_removed: bool
    row_issue_count: int
    row_issues: list[dict[str, Any]]
    row_issues_truncated: bool
    missing_value_counts: dict[str, int]
    warnings: list[str]
