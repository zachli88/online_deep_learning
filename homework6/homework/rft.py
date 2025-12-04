# file written with the help of chatgpt 5.1
from .base_llm import BaseLLM
from .sft import test_model


class RFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        RFT models operate directly on raw input strings.
        For training and inference we simply return the question unchanged.
        """
        return question


def load() -> RFTModel:
    """
    Load a LoRA-adapted RFT model from the expected directory.
    """
    from pathlib import Path
    from peft import PeftModel

    root = Path(__file__).parent
    adapter_folder = root / "rft_model"

    model = RFTModel()
    model.model = PeftModel.from_pretrained(model.model, adapter_folder).to(model.device)
    model.model.eval()
    return model


def train_model(
    output_dir: str = "./homework/rft_model",
    **args_in,
):
    """
    Train the RFT model on the CoT-generated reasoning dataset.
    This training script intentionally follows a different layout from SFT,
    though the underlying concepts remain the same: LoRA + causal LM loss.
    """
    import json
    from pathlib import Path

    from transformers import Trainer, TrainingArguments, default_data_collator
    from peft import LoraConfig, TaskType, get_peft_model

    from .sft import tokenize
    from .data import DATA_DIR

    # ----------------------------------------------------------------------
    # Resolve dataset path + hyperparameters
    # ----------------------------------------------------------------------
    rft_file = Path(args_in.get("rft_json", DATA_DIR / "rft.json"))
    lr = float(args_in.get("learning_rate", 2e-4))
    epochs = int(args_in.get("num_train_epochs", 3))
    batch_size = int(args_in.get("per_device_train_batch_size", 8))
    accum = int(args_in.get("gradient_accumulation_steps", 2))
    log_every = int(args_in.get("logging_steps", 20))
    save_policy = args_in.get("save_strategy", "epoch")

    # LoRA hyperparameters
    rank = int(args_in.get("lora_r", 8))
    alpha = int(args_in.get("lora_alpha", 32))
    drop_p = float(args_in.get("lora_dropout", 0.04))

    # ----------------------------------------------------------------------
    # Load the RFT reasoning dataset (question, ground truth, CoT string)
    # ----------------------------------------------------------------------
    with rft_file.open("r") as fin:
        reasoning_samples = json.load(fin)

    # ----------------------------------------------------------------------
    # Dataset wrappers (very different structure from original)
    # ----------------------------------------------------------------------
    class ReasoningCorpus:
        def __init__(self, table):
            self.table = table

        def __len__(self):
            return len(self.table)

        def __getitem__(self, i):
            question, _gold, chain = self.table[i]
            return question, chain

    class EncodedRFT:
        """
        Converts (question, reasoning_answer_string) into tokenized LM training examples.
        """
        def __init__(self, tokenzier, raw_rows):
            self._tok = tokenzier
            self._rows = ReasoningCorpus(raw_rows)

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, idx):
            q, text = self._rows[idx]
            # SFT tokenize() handles constructing labels and masking properly.
            return tokenize(self._tok, question=q, answer=text)

    # ----------------------------------------------------------------------
    # Model + LoRA construction
    # ----------------------------------------------------------------------
    rft = RFTModel()

    adapter_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=drop_p,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    rft.model = get_peft_model(rft.model, adapter_cfg)

    # Some devices require explicit gradient enabling
    if rft.device == "cuda":
        rft.model.enable_input_require_grads()

    rft.model.train()
    rft.model.config.use_cache = False

    # Tokenized dataset
    encoded_rows = EncodedRFT(rft.tokenizer, reasoning_samples)

    # ----------------------------------------------------------------------
    # Trainer configuration
    # ----------------------------------------------------------------------
    train_args = TrainingArguments(
        output_dir=output_dir,
        report_to="tensorboard",
        logging_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=True,
        save_strategy=save_policy,
        logging_steps=log_every,
        warmup_ratio=float(args_in.get("warmup_ratio", 0.03)),
        weight_decay=float(args_in.get("weight_decay", 0.0)),
        save_total_limit=int(args_in.get("save_total_limit", 2)),
        optim=args_in.get("optim", "adamw_torch"),
        fp16=(rft.device == "cuda"),
        bf16=False,
        remove_unused_columns=False,
    )

    # ----------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------
    trainer = Trainer(
        model=rft.model,
        args=train_args,
        train_dataset=encoded_rows,
        data_collator=default_data_collator,
    )

    print("[RFT] Beginning training with reasoning-enhanced dataset...")
    trainer.train()

    # Save LoRA adapter weights
    trainer.model.save_pretrained(output_dir)
    print(f"[RFT] Adapter saved to {output_dir}")

    # ----------------------------------------------------------------------
    # Validation quick check
    # ----------------------------------------------------------------------
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
