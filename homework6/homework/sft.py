# file written with the help of chatgpt 5.1

from .base_llm import BaseLLM
from .data import Dataset, benchmark

class SFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        SFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        return question


def load() -> SFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full

def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    rounded = round(answer, 1)
    formatted = f"<answer>{rounded}</answer>"
    
    return {"question": prompt,"answer": formatted}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = "./homework/sft_model",
    **unused_args,
):
    """
    Supervised fine-tuning for numeric unit-conversion answers.
    Uses a lightweight LoRA adapter layered on top of SmolLM2.
    """
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model

    # ---------------------------------------------------------
    # Initialize a lightweight base model (no chat formatting)
    # ---------------------------------------------------------
    base = SFTModel()

    # ---------------------------------------------------------
    # LoRA adapter configuration
    # ---------------------------------------------------------
    adapter_cfg = LoraConfig(
        r=12,
        lora_alpha=36,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
    )

    # Inject adapters
    tuned_model = get_peft_model(base.model, adapter_cfg)
    tuned_model.enable_input_require_grads()
    tuned_model.print_trainable_parameters()

    # ---------------------------------------------------------
    # Data: convert the raw (prompt, answer) pairs into tokens
    # ---------------------------------------------------------
    raw_split = Dataset("train")
    tokenized_train = TokenizedDataset(
        tokenizer=base.tokenizer,
        data=raw_split,
        format_fn=format_example,
    )

    # ---------------------------------------------------------
    # Training configuration for HuggingFace Trainer
    # ---------------------------------------------------------
    train_cfg = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=10,
        gradient_checkpointing=False,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        weight_decay=0.0,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=output_dir,
        report_to="tensorboard",
        remove_unused_columns=False,
        fp16=False,
        max_grad_norm=1.0,
    )

    # ---------------------------------------------------------
    # Launch training
    # ---------------------------------------------------------
    trainer = Trainer(
        model=tuned_model,
        args=train_cfg,
        train_dataset=tokenized_train,
    )

    print("Beginning supervised fine-tuning...")
    trainer.train()

    # ---------------------------------------------------------
    # Save adapter weights for grader consumption
    # ---------------------------------------------------------
    trainer.save_model(output_dir)
    print(f"Model adapters saved to: {output_dir}")

    # ---------------------------------------------------------
    # Optional smoke test
    # ---------------------------------------------------------
    print("Evaluating saved SFT model...")
    # test_model(output_dir)



def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTModel()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
