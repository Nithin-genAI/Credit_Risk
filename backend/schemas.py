from typing import Optional
from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    """
    Raw applicant features expected by the trained model.

    The frontend will eventually organize these fields into
    user-friendly sections.
    """

    application_id: str = Field(..., description="Bank application ID")
    name: str = Field(..., description="Applicant name")

    # Financial
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None

    # Family / demographics
    CNT_CHILDREN: int = 0
    CNT_FAM_MEMBERS: Optional[float] = None
    DAYS_BIRTH: float
    DAYS_EMPLOYED: Optional[float] = None

    # External credit indicators
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None

    # Contract
    NAME_CONTRACT_TYPE: Optional[str] = None

    # Other fields can be added as we expose them
    # in the bank-facing application form.