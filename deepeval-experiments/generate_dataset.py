import os
from pathlib import Path

from deepeval.models import OpenAIEmbeddingModel, OpenAIModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, StylingConfig
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def build_models() -> tuple[OpenAIModel, OpenAIEmbeddingModel]:
    """
    ```.env
    AZURE_OPENAI_ENDPOINT=https://<name>.services.ai.azure.com/openai/v1/responses
    AZURE_OPENAI_MODEL=gpt-4.1-mini
    AZURE_EMBEDDING_MODEL_NAME=text-embedding-3-small
    AZURE_OPENAI_API_KEY=<api-key>
    ```
    """
    load_dotenv(ROOT / ".env")
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    base_url = endpoint.removesuffix("/responses") + "/"

    chat = OpenAIModel(
        model=os.environ["AZURE_OPENAI_MODEL"],
        api_key=api_key,
        base_url=base_url,
    )
    embedder = OpenAIEmbeddingModel(
        model=os.environ["AZURE_EMBEDDING_MODEL_NAME"],
        api_key=api_key,
        base_url=base_url,
    )
    return chat, embedder


def main() -> None:
    chat, embedder = build_models()

    synthesizer = Synthesizer(
        model=chat,
        styling_config=StylingConfig(
            scenario="Readers asking questions about Oscar Wilde's The Picture of Dorian Gray",
            task="Answer accurately using only the provided novel, distinguishing events and details stated in the text from interpretation",
            input_format="Natural-language questions about characters, events, relationships, themes, settings, symbols, and quotations",
            expected_output_format="A concise, precise answer grounded in the novel, with the relevant chapter or passage when available",
        ),
        cost_tracking=True,
    )

    synthesizer.generate_goldens_from_docs(
        document_paths=[str(ROOT / "docs" / "the_picture_of_dorian_gray.pdf")],
        max_goldens_per_context=2,
        context_construction_config=ContextConstructionConfig(
            embedder=embedder,
            critic_model=chat,
            max_contexts_per_document=20,
            context_quality_threshold=0.0,
        ),
    )
    print(f"Generation cost: ${synthesizer.synthesis_cost}")

    output_dir = ROOT / "generated_test_sets"
    output_dir.mkdir(exist_ok=True)
    output_path = synthesizer.save_as(
        file_type="json",
        directory=str(output_dir),
        file_name="dorian_gray_single_turn",
    )
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()