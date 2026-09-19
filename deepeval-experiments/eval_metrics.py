import os
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def build_model() -> OpenAIModel:
    """
    ```.env
    AZURE_OPENAI_ENDPOINT=https://<name>.services.ai.azure.com/openai/v1/responses
    AZURE_OPENAI_MODEL=gpt-4.1-mini
    AZURE_OPENAI_API_KEY=<api-key>
    ```
    """
    load_dotenv(ROOT / ".env")
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") 
    base_url = endpoint.removesuffix("/responses") + "/"
    return OpenAIModel(
        model=os.environ["AZURE_OPENAI_MODEL"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=base_url,
        temperature=float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0")),
    )


def testing_contextual_relevancy(eval_model: OpenAIModel) -> None:
    print("Evaluating metric: Contextual Relevancy")
    input_text = "Which bird is widely recognized as the first scientifically documented poisonous bird?"
    test_case_contextual_relevancy_pass = LLMTestCase(
        input=input_text,
        retrieval_context=[
            "The hooded pitohui (Pitohui dichrous) is a bird native to New Guinea.",
            "Researchers reported in the 1980s that the hooded pitohui's skin and feathers contain batrachotoxins, potent neurotoxins.",
            "Because of these toxins, the hooded pitohui is widely described as the first scientifically documented poisonous bird.",
        ],
    )
    test_case_contextual_relevancy_fail = LLMTestCase(
        input=input_text,
        retrieval_context=[
            "The emperor penguin is the tallest and heaviest living penguin species.",
            "Penguins are flightless seabirds that primarily inhabit the Southern Hemisphere.",
            "The bald eagle is a large bird of prey native to North America.",
        ],
    )
    contextual_relevancy = ContextualRelevancyMetric(
        model=eval_model, verbose_mode=True
    )
    evaluate(
        [test_case_contextual_relevancy_pass, test_case_contextual_relevancy_fail],
        [contextual_relevancy],
        display_config=DisplayConfig(
            verbose_mode=True,
            truncate_passing_cases=False,
            file_type="md",
            file_output_dir=str(ROOT / "results"),
            results_folder=str(ROOT / "results"),
        ),
    )


def testing_contextual_precision(eval_model: OpenAIModel) -> None:
    print("Evaluating metric: Contextual Precision")
    input_text = "Why do wombats produce cube-shaped poop instead of round pellets?"
    expected_output = (
        "Wombat cubes are formed in the lower intestine, where regions with "
        "different elasticity shape the feces as it dries. The cubes are "
        "useful for territorial marking because they are less likely to roll "
        "away from rocks and logs."
    )

    test_case_contextual_precision_pass = LLMTestCase(
        input=input_text,
        expected_output=expected_output,
        retrieval_context=[
            ("Wombat feces become cube-shaped in the lower intestine because "
            "adjacent regions of the intestinal wall have different "
            "elasticity and contract at different rates."),
            ("Wombats use their feces for territorial marking, and cube-shaped "
            "droppings are less likely to roll off rocks or logs."),
            ("Wombats are marsupials native to Australia and are primarily "
            "nocturnal herbivores."),
        ],
    )
    test_case_contextual_precision_fail = LLMTestCase(
        input=input_text,
        expected_output=expected_output,
        retrieval_context=[
            ("The platypus is a mammal that lays eggs and has electroreception "
            "in its bill."),
            ("Wombats are marsupials native to Australia and are primarily "
            "nocturnal herbivores."),
            ("Wombat feces become cube-shaped in the lower intestine because "
            "adjacent regions of the intestinal wall have different "
            "elasticity and contract at different rates."),
        ],
    )
    contextual_precision = ContextualPrecisionMetric(
        model=eval_model,
        verbose_mode=True,
    )
    evaluate(
        [test_case_contextual_precision_pass, test_case_contextual_precision_fail],
        [contextual_precision],
        display_config=DisplayConfig(
            verbose_mode=True,
            truncate_passing_cases=False,
            file_type="md",
            file_output_dir=str(ROOT / "results"),
            results_folder=str(ROOT / "results"),
        ),
    )


def testing_contextual_recall(eval_model: OpenAIModel) -> None:
    print("Evaluating metric: Contextual Recall")
    input_text = ""  # Why can axolotls regrow a limb and remain juvenile-looking?"
    expected_output = (
        "Axolotls retain juvenile features because they undergo neoteny and "
        "reach adulthood without completing metamorphosis. They can regrow "
        "limbs because their cells form a blastema at the injury site, and "
        "they can regenerate tissues including parts of the spinal cord, "
        "heart, and brain."
    )

    test_case_contextual_recall_pass = LLMTestCase(
        input=input_text, # input is not really used in the Contextual Recall metric
        expected_output=expected_output,
        retrieval_context=[
            ("Axolotls exhibit neoteny: they become sexually mature while "
            "retaining juvenile traits and usually do not complete "
            "metamorphosis into a land-dwelling adult form."),
            ("After an injury, axolotls form a blastema, a mass of cells that "
            "helps rebuild a lost limb."),
            ("Axolotls can regenerate several tissues, including parts of the "
            "spinal cord, heart, and brain."),
        ],
    )
    test_case_contextual_recall_fail = LLMTestCase(
        input=input_text, # input is not really used in the Contextual Recall metric
        expected_output=expected_output,
        retrieval_context=[
            ("The Eiffel Tower was initially built as the entrance arch for "
            "the 1889 World's Fair in Paris."),
            ("A standard QWERTY keyboard places the letters Q, W, and E next "
            "to one another on its top row."),
        ],
    )
    contextual_recall = ContextualRecallMetric(
        model=eval_model,
        verbose_mode=True,
    )
    evaluate(
        [test_case_contextual_recall_pass, test_case_contextual_recall_fail],
        [contextual_recall],
        display_config=DisplayConfig(
            verbose_mode=True,
            truncate_passing_cases=False,
            file_type="md",
            file_output_dir=str(ROOT / "results"),
            results_folder=str(ROOT / "results"),
        ),
    )


def testing_faithfulness(eval_model: OpenAIModel) -> None:
    print("Evaluating metric: Faithfulness")
    input_text = "How can a gecko walk across a ceiling without falling?"
    retrieval_context = [
        ("Gecko toes are covered with millions of microscopic hair-like setae, "
        "which branch into even smaller structures called spatulae."),
        ("The spatulae make close contact with surfaces and create weak van der "
        "Waals interactions that let geckos adhere to walls and ceilings."),
        ("Geckos can detach their feet by changing the angle of their toes, "
        "so their adhesion does not rely on glue or suction cups."),
    ]
    test_case_faithfulness_pass = LLMTestCase(
        input=input_text,
        actual_output=(
            "A gecko's toes have millions of microscopic setae that branch "
            "into spatulae. These structures make close contact with a surface "
            "and create weak van der Waals interactions, allowing the gecko "
            "to adhere to ceilings without glue or suction cups."
        ),
        retrieval_context=retrieval_context,
    )
    test_case_faithfulness_fail = LLMTestCase(
        input=input_text,
        actual_output=(
            "Geckos stay on ceilings because they have smooth, clawless feet "
            "that act like magnetic suction cups. They use a thick layer of "
            "glue to attach themselves to the ceiling, and their toes do not "
            "contain setae or spatulae."
        ),
        retrieval_context=retrieval_context,
    )
    faithfulness = FaithfulnessMetric(
        model=eval_model,
        verbose_mode=True,
    )
    evaluate(
        [test_case_faithfulness_pass, test_case_faithfulness_fail],
        [faithfulness],
        display_config=DisplayConfig(
            verbose_mode=True,
            truncate_passing_cases=False,
            file_type="md",
            file_output_dir=str(ROOT / "results"),
            results_folder=str(ROOT / "results"),
        ),
    )


def testing_answer_relevancy(eval_model: OpenAIModel) -> None:
    print("Evaluating metric: Answer Relevancy")
    input_text = "How can a gecko walk across a ceiling without falling?"
    test_case_answer_relevancy_pass = LLMTestCase(
        input=input_text,
        actual_output=(
            "A gecko's microscopic setae branch into spatulae that make close "
            "contact with the ceiling and create weak van der Waals "
            "interactions. This lets the gecko adhere to the ceiling without "
            "glue or suction cups."
        ),
    )
    test_case_answer_relevancy_fail = LLMTestCase(
        input=input_text,
        actual_output=(
            "Geckos are small reptiles found in warm regions, and many species "
            "are active at night. Their eyes are often large, and some geckos "
            "can vocalize with chirps and clicks."
        ),
    )
    answer_relevancy = AnswerRelevancyMetric(
        model=eval_model,
        verbose_mode=True,
    )
    evaluate(
        [test_case_answer_relevancy_pass, test_case_answer_relevancy_fail],
        [answer_relevancy],
        display_config=DisplayConfig(
            verbose_mode=True,
            truncate_passing_cases=False,
            file_type="md",
            file_output_dir=str(ROOT / "results"),
            results_folder=str(ROOT / "results"),
        ),
    )


def main():
    eval_model = build_model()

    # Retrieval metrics
    testing_contextual_relevancy(eval_model)
    testing_contextual_precision(eval_model)
    testing_contextual_recall(eval_model)

    # Generation metrics
    testing_faithfulness(eval_model)
    testing_answer_relevancy(eval_model)


if __name__ == "__main__":
    main()
