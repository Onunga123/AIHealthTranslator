# backend/training/train_luo.py
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
import evaluate

# -----------------------------
# 1. Dataset Path
# -----------------------------
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "luo_dataset_clean.csv"))
print(f"📂 Loading dataset from: {DATA_PATH}")

dataset = load_dataset("csv", data_files={"full": DATA_PATH})["full"].train_test_split(test_size=0.1)
# No renaming needed if already lowercase
if "English" in dataset["train"].column_names:
    dataset = dataset.rename_column("English", "english")
if "Luo" in dataset["train"].column_names:
    dataset = dataset.rename_column("Luo", "luo")

# -----------------------------
# 2. Model & Tokenizer
# -----------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-en-swc"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

# Force CPU (Intel GPU not supported)
device = torch.device("cpu")
model = model.to(device)
print(f"🖥 Training will run on: {device}")

# -----------------------------
# 3. Preprocessing
# -----------------------------
def preprocess_function(examples):
    inputs = examples["english"]
    targets = examples["luo"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# -----------------------------
# 4. Training Arguments
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./luo_model",
    eval_strategy="epoch",          # compatible with your transformers version
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # small for CPU
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,            # 🔹 increased to 10
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="epoch",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -----------------------------
# 5. Metrics
# -----------------------------
sacrebleu = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = sacrebleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": bleu["score"]}

# -----------------------------
# 6. Callback for live translations
# -----------------------------
class TranslationCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        sample_text = "Hello, how are you?"
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model.generate(**inputs, max_length=40)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n📝 Epoch {int(state.epoch)} sample translation:")
        print(f"  EN: {sample_text}")
        print(f"  LUO: {translation}\n")

# -----------------------------
# 7. Trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TranslationCallback()],   # ✅ live feedback
)

# -----------------------------
# 8. Train + Save + Test
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete!")

    # Save model & tokenizer
    save_path = "./luo_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"💾 Model saved to: {save_path}")

    # Final test translation
    test_sentence = "The hospital is very far from the village."
    inputs = tokenizer(test_sentence, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=40)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📝 Final test translation:\n  EN: {test_sentence}\n  LUO: {translation}")
