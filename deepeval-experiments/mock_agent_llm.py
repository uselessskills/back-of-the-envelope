import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent


def build_client() -> tuple[OpenAI, str]:
    load_dotenv(ROOT / ".env")
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    base_url = endpoint.removesuffix("/responses") + "/"
    client = OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=base_url,
    )
    return client, os.environ["AZURE_OPENAI_MODEL"]


def mock_agent(client: OpenAI, model: str, user_input: str, context: list[str]) -> str:
    context_text = "\n\n".join(context)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compliance assistant answering questions about the SAMA document. "
                    "Answer only from the supplied context. If the context does not support an answer, "
                    "say that the document does not provide enough information. Be concise and precise."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion:\n{user_input}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM to mock responses for a DeepEval dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "generated_test_sets" / "dorian_gray_single_turn.json",
        help="Generated DeepEval goldens JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "generated_test_sets" / "dorian_gray_single_turn_llm_mocked.json",
        help="Output JSON file containing LLM-generated actual_output values.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N records.")
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    records = records[: args.limit] if args.limit is not None else records
    client, model = build_client()

    for index, record in enumerate(records, start=1):
        record["actual_output"] = mock_agent(
            client,
            model,
            record["input"],
            record.get("context", []),
        )
        print(f"Generated response {index}/{len(records)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} LLM-mocked responses to {args.output}")


if __name__ == "__main__":
    main()