import json
from colorama import Fore


def preprocess(input_path: str = "tm1data.json", output_path: str = "data/instruction.json") -> None:
    """
    Flatten the raw synthetic dataset into a flat list of {question, answer} pairs
    and write them to the output JSON file.

    Args:
        input_path: Path to the raw generated dataset (output of syntheticdatageneration.py).
        output_path: Destination path for the flat instruction dataset.
    """
    instructions = []

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, chunk in data.items():
        for pair in chunk["generated"]:
            instructions.append(
                {
                    "question": pair["question"],
                    "answer": pair["answer"],
                }
            )
        print(Fore.YELLOW + str(chunk) + Fore.RESET)
        print("\n~~~~~~~~~~~~~~~~~~~~~")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=2)

    print(Fore.GREEN + f"\n[Preprocessing] Saved {len(instructions)} pairs → {output_path}" + Fore.RESET)

    # Quick sanity check — print the first 10 records
    with open(output_path, "r", encoding="utf-8") as f:
        preview = json.load(f)
    print(Fore.LIGHTMAGENTA_EX + str(preview[:10]) + Fore.RESET)


if __name__ == "__main__":
    preprocess()
