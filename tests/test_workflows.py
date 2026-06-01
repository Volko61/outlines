import json
from typing import Literal

import pytest
from pydantic import BaseModel, Field
import transformers

import outlines
from outlines.workflows import TwoStepJsonGenerator


TEST_MODEL = "erwanf/gpt2-mini"


@pytest.fixture
def model():
    return outlines.from_transformers(
        transformers.AutoModelForCausalLM.from_pretrained(TEST_MODEL),
        transformers.AutoTokenizer.from_pretrained(TEST_MODEL),
    )


def test_two_step_json_generator(model):
    class Sentiment(BaseModel):
        sentiment: Literal["positive", "neutral", "negative"]
        summary: str = Field(max_length=10)

    generator = TwoStepJsonGenerator(
        model,
        Sentiment,
        enum_field="sentiment",
        summary_field="summary",
    )
    result = generator(
        "J'adore le beau temps",
        stage2_kwargs={"max_new_tokens": 3},
    )
    payload = json.loads(result)
    assert payload["sentiment"] in ["positive", "neutral", "negative"]
    assert isinstance(payload["summary"], str)


def test_two_step_json_missing_enum(model):
    class Sentiment(BaseModel):
        sentiment: str
        summary: str = Field(max_length=10)

    with pytest.raises(ValueError):
        TwoStepJsonGenerator(
            model,
            Sentiment,
            enum_field="sentiment",
            summary_field="summary",
        )


def test_two_step_json_missing_max_length(model):
    class Sentiment(BaseModel):
        sentiment: Literal["positive", "neutral"]
        summary: str

    with pytest.raises(ValueError):
        TwoStepJsonGenerator(
            model,
            Sentiment,
            enum_field="sentiment",
            summary_field="summary",
        )
