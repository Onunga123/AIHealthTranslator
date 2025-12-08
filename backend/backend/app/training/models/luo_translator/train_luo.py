
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import load_dataset
import os
import sys

MODEL_NAME = os.path.join(os.path.dirname(__file__))
DATA_TRAIN = "train.csv"
DATA_TEST = "test.csv"

def print_error(msg):
    print(f"\n[ERROR] {msg}\n", file=sys.stderr)

def safe_call(func, desc):
    try:
        print(f"➡️ {desc}...")
        return func()
    except Exception as e:
        print_error(f"Failed during: {desc}\n{type(e).__name__}: {e}")
        sys.exit(1)

# 1. Load and prepare the dataset
def load_and_prepare_dataset():
    def _load():
        return load_dataset("csv", data_files={"train": DATA_TRAIN, "test": DATA_TEST}, delimiter=",")
    dataset = safe_call(_load, f"Loading dataset from {DATA_TRAIN} and {DATA_TEST}")
    dataset = safe_call(lambda: dataset.rename_columns({"translation_en": "en_clean", "translation_romance": "luo_clean"}), "Renaming columns")
    dataset = safe_call(lambda: dataset.filter(lambda x: x["en_clean"] and x["luo_clean"]), "Filtering empty rows")
    dataset = safe_call(lambda: dataset.train_test_split(test_size=0.2), "Splitting train/test")
    return dataset

def load_tokenizer_and_model():
    print(f"🔄 Loading local Luo model from: {MODEL_NAME}")
    tokenizer = safe_call(lambda: AutoTokenizer.from_pretrained(MODEL_NAME), f"Loading local tokenizer from {MODEL_NAME}")
    model = safe_call(lambda: AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME), f"Loading local model from {MODEL_NAME}")
    return tokenizer, model

def preprocess_function(examples, tokenizer):
    inputs = examples["en_clean"]
    targets = examples["luo_clean"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("� Starting Luo model training script...")
    dataset = load_and_prepare_dataset()
    print(f"✅ Training dataset size: {len(dataset['train'])}")
    print(f"✅ Validation dataset size: {len(dataset['test'])}")

    tokenizer, model = load_tokenizer_and_model()

    print("�🔄 Tokenizing dataset...")
    try:
        tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    except Exception as e:
        print_error(f"Tokenization failed: {type(e).__name__}: {e}")
        sys.exit(1)

    OUTPUT_DIR = os.path.join("backend", "app", "training", "models", "luo_translator")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=2,
        predict_with_generate=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_strategy="steps",
        logging_steps=10,
        save_safetensors=False,
    )

    print("🧮 Setting up data collator...")
    try:
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    except Exception as e:
        print_error(f"Data collator setup failed: {type(e).__name__}: {e}")
        sys.exit(1)

    print("🛠️ Setting up trainer...")
    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    except Exception as e:
        print_error(f"Trainer setup failed: {type(e).__name__}: {e}")
        sys.exit(1)

    print("🚀 Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print_error(f"Training failed: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"💾 Saving trained Luo model to {OUTPUT_DIR}...")
    try:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    except Exception as e:
        print_error(f"Model saving failed: {type(e).__name__}: {e}")
        sys.exit(1)

    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    print("🔍 Verifying saved files in:", OUTPUT_DIR)
    for file in required_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        if os.path.exists(file_path):
            print(f"✅ {file} exists")
        else:
            print(f"❌ Missing: {file}")

    print("🎉 Training complete! Model is ready to be used in the API.")

if __name__ == "__main__":
    main()
