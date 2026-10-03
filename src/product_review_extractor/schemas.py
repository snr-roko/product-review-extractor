from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class Sentiment(str, Enum):
    delighted = "delighted"
    satisfied = "satisfied"
    neutral = "neutral"
    disappointed = "disappointed"
    frustrated = "frustrated"
    angry = "angry"

class RepurchaseIntent(str, Enum):
    yes = "yes"
    no = "no"
    maybe = "maybe"
    unclear = "unclear"


class ProductReviewSchema(BaseModel):
    product_name: str = Field(min_length=1)
    price: float | None = Field(ge=0)
    currency: str | None = Field(min_length=1)
    sentiment: Sentiment
    repurchase_intent: RepurchaseIntent
    intent_evidence: str | None = Field(min_length=1)
    issues_mentioned: list[str]

    @model_validator(mode="after")
    def currency_requires_price(self) -> Self:
        if self.currency is not None and self.price is None:
            raise ValueError("currency must be null when price is null")
        return self