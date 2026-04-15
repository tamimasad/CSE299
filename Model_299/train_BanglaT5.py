
import os
import numpy as np
import torch
import pandas as pd
import nltk
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# ── NLTK
nltk.download("punkt",   quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# 1.  LOAD DATA  (keep dialect_label for dialect-aware prefix)


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df[['text', 'standard', 'dialect_label']].dropna(
        subset=['text', 'standard'])
    return Dataset.from_pandas(df, preserve_index=False)


train_dataset = load_data('./data/processed/cleaned_train.csv')
val_dataset = load_data('./data/processed/cleaned_val.csv')
print(f"Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}")

# 2.  TOKENISER & MODEL

MODEL_NAME = "csebuetnlp/banglat5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model = model.to(device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 3.  PREPROCESSING  — dialect-aware prefix

def preprocess_function(examples):
    inputs = []
    for text, dialect in zip(examples["text"], examples["dialect_label"]):
        prefix = f"translate {dialect} dialect to standard Bangla: "
        inputs.append(prefix + str(text))

    targets = [str(t) for t in examples["standard"]]

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=128,
        truncation=True,
        padding=False,
    )
    return model_inputs


print("Tokenising...")
tokenized_train = train_dataset.map(
    preprocess_function, batched=True,
    remove_columns=train_dataset.column_names,
    desc="Train"
)
tokenized_val = val_dataset.map(
    preprocess_function, batched=True,
    remove_columns=val_dataset.column_names,
    desc="Val"
)
# 4.  METRICS
metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")
metric_meteor = evaluate.load("meteor")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds >= 0, preds,  tokenizer.pad_token_id)
    labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)

    decoded_preds = [p.strip() for p in tokenizer.batch_decode(
        preds,  skip_special_tokens=True)]
    decoded_labels = [l.strip() for l in tokenizer.batch_decode(
        labels, skip_special_tokens=True)]

    bleu = metric_bleu.compute(predictions=decoded_preds,
                               references=[[l] for l in decoded_labels])
    chrf = metric_chrf.compute(predictions=decoded_preds,
                               references=[[l] for l in decoded_labels])
    meteor = metric_meteor.compute(predictions=decoded_preds,
                                   references=decoded_labels)
    return {
        "bleu":   round(bleu["score"],    2),
        "chrf":   round(chrf["score"],    2),
        "meteor": round(meteor["meteor"], 4),
    }


# 5.  TRAINING ARGUMENTS
BATCH_SIZE = 8
GRADIENT_ACCUM_STEPS = 4      
NUM_EPOCHS = 10
TOTAL_STEPS = (len(tokenized_train) // (BATCH_SIZE *
               GRADIENT_ACCUM_STEPS)) * NUM_EPOCHS
WARMUP_STEPS = max(1, int(TOTAL_STEPS * 0.06))

print(f"Effective batch size : {BATCH_SIZE * GRADIENT_ACCUM_STEPS}")
print(
    f"Total steps          : {TOTAL_STEPS:,}  |  Warmup steps: {WARMUP_STEPS}")

training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_banglat5",

    # Batch & accumulation
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,  # IMPROVEMENT 3

    # Optimiser
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    save_total_limit=3,
    # Generation during eval
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    # Regularisation
    label_smoothing_factor=0.1,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    logging_steps=50,
    report_to="none",
)
# 6.  TRAINER
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    ),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3)],
)

# 7.  TRAIN
print("\nStarting training...")
trainer.train()

print("\nSaving model...")
trainer.save_model("./fine_tuned_banglat5_final")
tokenizer.save_pretrained("./fine_tuned_banglat5_final")

pd.DataFrame(trainer.state.log_history).to_csv(
    "./fine_tuned_banglat5_final/training_log.csv", index=False
)

print("Done! Model saved to ./fine_tuned_banglat5_final_new")
