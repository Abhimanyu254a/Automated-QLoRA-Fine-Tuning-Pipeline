import json
from typing import List

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore
from pydantic import BaseModel
from litellm import completion

from generated_prompt import prompt_template


class Record(BaseModel):
    question: str
    answer: str


class Response(BaseModel):
    generated: List[Record]


def llm_call(data: str, num_records: int = 5) -> dict:
    """
    Call the local Ollama LLM to generate Q&A pairs for a text chunk.

    Args:
        data: Contextualised text chunk.
        num_records: Number of Q&A pairs to generate.

    Returns:
        Parsed JSON dict containing the 'generated' list of Q&A pairs.
    """
    stream = completion(
        model="ollama_chat/qwen2.5:14b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
    )

    response_text = ""
    for x in stream:
        delta = x["choices"][0]["delta"]["content"]
        if delta is not None:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            response_text += delta

    return json.loads(response_text)


if __name__ == "__main__":
    converter = DocumentConverter()
    doc = converter.convert("tm1_dg_dvlpr-10pages.pdf").document
    chunker = HybridChunker()
    chunks = list(chunker.chunk(dl_doc=doc))

    total_chunks = len(chunks)
    print(Fore.CYAN + f"[DataGen] Processing {total_chunks} chunks ..." + Fore.RESET)

    dataset = {}
    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"\n[Chunk {i + 1}/{total_chunks}] Raw text preview:" + Fore.RESET)
        print(Fore.YELLOW + f"{chunk.text[:300]}…" + Fore.RESET)

        enriched_text = chunker.contextualize(chunk=chunk)
        print(Fore.LIGHTMAGENTA_EX + f"Contextualized:\n{enriched_text[:300]}…" + Fore.RESET)

        try:
            data = llm_call(enriched_text)
            dataset[i] = {"generated": data["generated"], "context": enriched_text}
        except Exception as exc:
            print(Fore.RED + f"[Chunk {i + 1}] LLM call failed — skipping. Error: {exc}" + Fore.RESET)
            continue

    with open("tm1data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(Fore.GREEN + f"\n[DataGen] Done! Saved {len(dataset)} chunks to tm1data.json" + Fore.RESET)
