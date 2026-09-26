import json
from pydantic import BaseModel
from litellm import completion
from colorama import Fore


class Score(BaseModel):
    score: int
    explanation: str


class Rank(BaseModel):
    accuracy: Score
    style: Score


def llm_call(record: str) -> dict:
    """
    Ask the LLM to score a single Q&A instruction pair on accuracy (1-10) and style (1-10).

    Scoring rules:
      - accuracy < 6  → pair is factually unreliable or not a real question
      - style < 6     → pair is harmful, unhelpful, or not honest
      - Either blank/ellipsis → score of 1 for both

    Args:
        record: A stringified Q&A pair dict.

    Returns:
        A dict with 'accuracy' and 'style' keys, each containing 'score' and 'explanation'.
    """
    stream = completion(
        model="ollama_chat/qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify this instruction tuning record between 1-10 in terms of accuracy "
                    "and style, provide explanations as to your reasoning. "
                    "The answer should be completely self contained. "
                    "If the question is not in fact a question it should receive 0 for accuracy. "
                    "If the answer does not adequately answer the question it should receive 1 for accuracy. "
                    "If the question or answer are in any way harmful, not helpful or not honest "
                    "it should be classified as 1 for style. "
                    "If the question or answer are blank or ..., return 1 for both.\n\n"
                    f"Record: {record}\n\nReturn as JSON"
                ),
            }
        ],
        stream=True,
        options={"num_predict": 2000, "temperature": 0.2},
        format=Rank.model_json_schema(),
    )

    data = ""
    for x in stream:
        delta = x["choices"][0]["delta"]["content"]
        if delta is not None:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            data += delta

    return json.loads(data)


if __name__ == "__main__":
    quality = []
    instructions = []

    with open("data/instruction.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(Fore.CYAN + f"[Quality] Scoring {total} instruction pairs ..." + Fore.RESET)

    for i, pair in enumerate(data):
        print(Fore.YELLOW + f"\n[{i + 1}/{total}] {pair}" + Fore.RESET)

        result = llm_call(str(pair))

        acc = result["accuracy"]["score"]
        sty = result["style"]["score"]
        print(
            Fore.CYAN
            + f"  → accuracy={acc}, style={sty}"
            + Fore.RESET
        )

        if acc >= 6 and sty >= 6:
            instructions.append(pair)
            quality.append({**pair, "quality": result})

    print(
        Fore.GREEN
        + f"\n[Quality] Kept {len(instructions)}/{total} high-quality pairs."
        + Fore.RESET
    )

    with open("data/instructionquality.json", "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=2)

    with open("qualityresults.json", "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    print(Fore.GREEN + "[Quality] Results saved." + Fore.RESET)