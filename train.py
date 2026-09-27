import os
import torch
from datasets import load_dataset
from colorama import Fore
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

# environment 
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", None)

NUM_PROC = int(os.environ.get("NUM_PROC", "4"))
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "qwen2.5-1.5b-SFT")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "50"))

# Dataset 
dataset = load_dataset("data", split="train")
print(Fore.YELLOW + f"[Dataset] Loaded {len(dataset)} samples" + Fore.RESET)
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)


# System prompt v3 
SYSTEM_PROMPT = (
    "You are a helpful, honest and harmless assistant designed to help engineers. "
    "Think through each question logically and provide an answer. "
    "Don't make things up; if you're unable to answer a question, advise the user "
    "that it is outside of your scope."
)

LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)


def format_chat_template(batch: dict, tokenizer) -> dict:
    """Convert raw Q&A pairs into Llama-3 chat-formatted strings."""
    questions = batch["question"]
    answers = batch["answer"]
    samples = []

    for question, answer in zip(questions, answers):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        samples.append(text)

    return {
        "instruction": questions,
        "response": answers,
        "text": samples,
    }


# Tokenizer form the transfomer 
print(Fore.CYAN + f"[Model] Loading tokenizer for {BASE_MODEL} ..." + Fore.RESET)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    token=HF_TOKEN,
)

train_dataset = dataset.map(
    lambda x: format_chat_template(x, tokenizer),
    num_proc=NUM_PROC,
    batched=True,
    batch_size=10,
)
print(Fore.LIGHTMAGENTA_EX + f"[Dataset] Sample after formatting:\n{train_dataset[0]}" + Fore.RESET)


# quantization QLoRA: 4-bit 
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Basic model 
print(Fore.CYAN + f"[Model] Loading {BASE_MODEL} in 4-bit ..." + Fore.RESET)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cuda:0",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    cache_dir="./workspace",
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA config 
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Training the Model 
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        dataset_text_field="text",
        max_seq_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",  # set to "wandb" or "tensorboard" if you want tracking
    ),
)

print(Fore.GREEN + "[Training] Starting fine-tuning ..." + Fore.RESET)
trainer.train()

print(Fore.GREEN + "[Training] Saving checkpoints ..." + Fore.RESET)
trainer.save_model("complete_checkpoint")
trainer.model.save_pretrained("final_model")
print(Fore.GREEN + "[Training] Done! Models saved to ./complete_checkpoint and ./final_model" + Fore.RESET)