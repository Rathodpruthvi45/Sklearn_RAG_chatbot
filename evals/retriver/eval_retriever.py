import json
import os

from deepeval.models import OllamaModel,GeminiModel
from deepeval.metrics import ContextualPrecisionMetric,ContextualRecallMetric,ContextualRelevancyMetric
from pathlib import Path
from src.retriever.retriever import retrieve
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import AsyncConfig
from deepeval import evaluate

# AQ.Ab8RN6LPWUeY6jVqBY8kOqeeFgNJQfnMtGGPGx43tREIUzSvSQ

os.environ[
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"
] = "600"

# client = OllamaModel(
#     model="qwen2.5:1.5b", base_url="http://localhost:11434", temperature=0
# )

client = GeminiModel(
     model="gemini-2.5-flash",api_key="AQ.Ab8RN6LPWUeY6jVqBY8kOqeeFgNJQfnMtGGPGx43tREIUzSvSQ", temperature=0
)


dataset_path=Path("data/retriever_golden_dataset.json")

with open(dataset_path,'r',encoding='utf-8') as file:
    dataset_raw=json.load(file)



metrics=[
    ContextualRecallMetric(model=client,include_reason=True,async_mode=False,threshold=0.7),
    ContextualPrecisionMetric(model=client,include_reason=True,async_mode=False,threshold=0.7),
    ContextualRelevancyMetric(model=client,include_reason=True,async_mode=False,threshold=0.7)

]

dataset = EvaluationDataset()


for golden in dataset_raw:
    question = golden["input"]
    expected_output = golden["expected_output"]

    retrieved_docs = retrieve(question)
    retrieval_context = [doc.page_content for doc in retrieved_docs]

    dataset.add_test_case(
        LLMTestCase(
            input=question,
            actual_output="Retrieval-only evaluation; no answer generated.",
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
    )

sample_words=dataset.test_cases[:15]



evaluate(sample_words, metrics,async_config=AsyncConfig(run_async=False))