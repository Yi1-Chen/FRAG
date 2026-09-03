"""Relearning stress-test: finetune unlearned model on forget data."""

import argparse
import os
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
try:
    import wandb
except ImportError:
    wandb = None


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "unlearning-pruning"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])
    args = parser.parse_args()

    seed_everything(args.seed)
    print(f"[seed] seed_everything({args.seed}) — cudnn.deterministic=True")

    # wandb setup
    report_to = "none"
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is required for --wandb. Install with: pip install wandb")
        model_short = args.model.split("/")[-1]
        run_name = args.wandb_run_name or f"relearn_{model_short}_ep{args.epochs}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=args.wandb_tags,
            config=vars(args),
        )
        report_to = "wandb"

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # No device_map: under accelerate launch (DDP), each rank places the model
    # on its assigned GPU via accelerator.prepare(). Single-process call still
    # works because Trainer moves the model to cuda before training.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print(f"Loading data: {args.data}")
    ds = load_dataset("text", data_files=args.data, split="train")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length, padding="max_length")

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        optim="adamw_bnb_8bit",
        eval_strategy="no",
        report_to=report_to,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("Starting relearning...")
    trainer.train()

    print(f"Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.wandb:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
