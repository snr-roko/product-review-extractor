import pytest
from pydantic import ValidationError

from product_review_extractor.schemas import ProductReviewSchema


def get_valid_review_data():
    return {
        "product_name": "Test Product",
        "price": "19.99",
        "currency": "USD",
        "sentiment": "satisfied",
        "repurchase_intent": "yes",
        "intent_evidence": "I would buy this again.",
        "issues_mentioned": [],
    }

def test_valid_object_passes():
    review = ProductReviewSchema.model_validate(get_valid_review_data())

    assert review.price == 19.99
    assert review.currency == "USD"

def test_missing_price_key_fails():
    data = get_valid_review_data()
    del data["price"]

    with pytest.raises(ValidationError) as exc:
        ProductReviewSchema.model_validate(data)

    assert exc.value.errors()[0]["loc"] == ("price",)

def test_nullable_fields_can_be_null():
    data = get_valid_review_data()
    data["price"] = None
    data["currency"] = None
    data["intent_evidence"] = None

    review = ProductReviewSchema.model_validate(data)

    assert review.price is None
    assert review.currency is None
    assert review.intent_evidence is None

def test_wrong_enum_value_fails():
    data = get_valid_review_data()
    data["sentiment"] = "happy"

    with pytest.raises(ValidationError) as exc:
        ProductReviewSchema.model_validate(data)

    assert exc.value.errors()[0]["loc"] == ("sentiment",)