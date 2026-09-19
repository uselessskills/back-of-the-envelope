import argparse
import json
import os
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import ErrorConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def build_model() -> OpenAIModel:
    load_dotenv(ROOT / ".env")
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    base_url = endpoint.removesuffix("/responses") + "/"
    return OpenAIModel(
        model=os.environ["AZURE_OPENAI_MODEL"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=base_url,
        temperature=float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0")),
    )


def load_test_cases(path: Path) -> list[LLMTestCase]:
    records = json.loads(path.read_text())
    return [
        LLMTestCase(
            input=record["input"],
            actual_output=record.get("actual_output") or "",
            expected_output=record["expected_output"],
            retrieval_context=record.get("context", []),
        )
        for record in records
    ]


def main() -> None:
    """
    >>> uv run python evaluate_dataset.py --dataset generated_test_sets/dorian_gray_single_turn.json
    """
    parser = argparse.ArgumentParser(
        description="Evaluate The Picture of Dorian Gray responses with DeepEval."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "generated_test_sets" / "dorian_gray_single_turn.json",
        help="JSON dataset containing generated questions, answers, and contexts.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="GEval passing threshold."
    )
    args = parser.parse_args()

    test_cases = load_test_cases(args.dataset)
    model = build_model()

    metrics = []
    if any(test_case.actual_output for test_case in test_cases):
        metrics.append(
            GEval(
                name="Dorian Gray Correctness",
                criteria=(
                    "Determine whether the actual output accurately answers the literary question using the expected output "
                    "and stays supported by the novel excerpts. Penalize unsupported claims, omissions, and contradictions."
                ),
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                    SingleTurnParams.RETRIEVAL_CONTEXT,
                ],
                threshold=args.threshold,
                model=model,
            )
        )

    contextual_precision = ContextualPrecisionMetric(model=model)
    contextual_recall = ContextualRecallMetric(model=model)
    contextual_relevancy = ContextualRelevancyMetric(model=model)
    metrics.extend([contextual_precision, contextual_recall, contextual_relevancy])

    evaluate(
        test_cases,
        metrics,
        error_config=ErrorConfig(ignore_errors=True),
    )
    print(f"Evaluated {len(test_cases)} test cases")


if __name__ == "__main__":
    main()
