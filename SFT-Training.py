import os
#os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ['WANDB_MODE'] = 'offline'

import sys
import logging
import argparse
import json

import torch
import datasets
import transformers
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import idr_torch  

wandb_project = "SFT-Qwen-4B-Instruct-Gen"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

logger = logging.getLogger(__name__)

def main():
    # CLI
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", type=str, help="HuggingFace model name")
    parser.add_argument("--path_train_dataset", type=str, default="./ft-data/data/train_data.json")
    parser.add_argument("--path_eval_dataset", type=str, default="./ft-data/data/test_data.json")
    parser.add_argument("--output_dir", type=str, default="./SFT-MistralNachos-models/")
    parser.add_argument("--logging_dir", type=str, default="./SFT-MistralNachos-logs/")
    parser.add_argument("--epochs", type=int, default=5, required=True)
    parser.add_argument("--batch_size", type=int, default=4, required=True)
    parser.add_argument("--save_steps", type=int, default=100, required=True)
    parser.add_argument("--logging_steps", type=int, default=10, required=True)
    parser.add_argument("--seed", type=int, default=42, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4, required=True)
    args = parser.parse_args()

    sft_config = SFTConfig(
        bf16=True,
        fp16=False,
        do_eval=True,
        eval_strategy="epoch",                 
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        push_to_hub=False,
        remove_unused_columns=True,
        report_to="wandb",
        save_strategy="steps",
        save_steps=args.save_steps,
        seed=args.seed,
        logging_dir=args.logging_dir,
        logging_first_step=True,
        group_by_length=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        local_rank=idr_torch.local_rank,     

        # ---- SFT-specific bits ----
        dataset_text_field="formatted_chat",
        max_length=2048,                      
        pad_to_multiple_of=8,                
    )

    set_seed(args.seed)

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = transformers.logging.get_verbosity()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load data
    with open(args.path_train_dataset, 'r') as f:
        train_data = json.load(f)
    with open(args.path_eval_dataset, 'r') as f:
        val_data = json.load(f)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"

    # Datasets
    train_dataset = Dataset.from_dict({"chat": train_data})
    eval_dataset  = Dataset.from_dict({"chat": val_data})

    train_dataset = train_dataset.map(
        lambda x: {"formatted_chat": tokenizer.apply_chat_template(
            x["chat"], tokenize=False, add_generation_prompt=False)}
    )
    eval_dataset = eval_dataset.map(
        lambda x: {"formatted_chat": tokenizer.apply_chat_template(
            x["chat"], tokenize=False, add_generation_prompt=False)}
    )

    # Model
    logger.info("*** Load base model ***")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=not sft_config.gradient_checkpointing,
    )
    # Resize embeddings & set pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

    # LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        modules_to_save=None,
        use_dora=True
    )

    # Trainer 
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    train_result = trainer.train()
    #train_result = trainer.train(resume_from_checkpoint=True)

    # Save
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(sft_config.output_dir)

    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(sft_config.output_dir)

if __name__ == "__main__":
    main()
