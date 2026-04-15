import pandas as pd
import torch
import os
import nltk
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import evaluate
from tqdm import tqdm

# Ensure NLTK data is available for METEOR
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# --- 1. Setup and Load Metrics ---
print("Loading evaluation metrics...")
sacrebleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
meteor = evaluate.load("meteor")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 2. Load Model ---
m2m100_path = "facebook/m2m100_418M"

print(f"Loading Fine-Tuned M2M100 from {m2m100_path}...")
tokenizer = M2M100Tokenizer.from_pretrained(m2m100_path)
model = M2M100ForConditionalGeneration.from_pretrained(m2m100_path).to(device)

model.eval()

tokenizer.src_lang = "bn"
BN_LANG_ID = tokenizer.get_lang_id("bn")

# --- 3. Generation & Evaluation Functions ---

BATCH_SIZE = 16


def generate_translations(texts, batch_size=BATCH_SIZE, desc="Translating"):
    """Translate a list of texts in batches."""
    all_predictions = []

    with tqdm(total=len(texts), desc=desc, unit="lines") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=BN_LANG_ID
                )

            decoded = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True)
            all_predictions.extend(decoded)
            pbar.update(len(batch))

    return all_predictions


def run_evaluation(source_texts, target_texts, dialect_name):
    predictions = generate_translations(
        source_texts,
        batch_size=BATCH_SIZE,
        desc=f"Processing {dialect_name}"
    )

    # Calculate Metrics
    bleu_result = sacrebleu.compute(
        predictions=predictions,
        references=[[t] for t in target_texts]
    )
    chrf_result = chrf.compute(
        predictions=predictions,
        references=[[t] for t in target_texts]
    )
    meteor_result = meteor.compute(
        predictions=predictions,
        references=target_texts
    )

    return {
        "Dialect": dialect_name,
        "BLEU": round(bleu_result["score"], 2),
        "chrF": round(chrf_result["score"], 2),
        "METEOR": round(meteor_result["meteor"], 4)
    }


# --- 4. Main Execution Loop ---
if __name__ == "__main__":
    dialects = ['Chittagong', 'Sylhet', 'Barishal', 'Noakhali', 'Mymensingh']
    vashantor_base = "./data/vashantor010"
    all_results = []

    for dialect in dialects:
        file_path = f"{vashantor_base}/{dialect} Test Translation.csv"

        if not os.path.exists(file_path):
            print(f"Skipping {dialect}: File not found at {file_path}")
            continue

        print(f"\n{'='*40}\nBenchmarking {dialect}\n{'='*40}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()

        std_col = 'bangla_speech'
        dia_col = next(
            (c for c in df.columns if dialect.lower() in c and 'bangla_speech' in c),
            None
        )

        if not dia_col or std_col not in df.columns:
            print(f"Skipping {dialect}: Required columns not found.")
            continue

        df_clean = df[[dia_col, std_col]].dropna()
        source = df_clean[dia_col].tolist()
        target = df_clean[std_col].tolist()

        results = run_evaluation(source, target, dialect)
        all_results.append(results)
        print(
            f"  BLEU: {results['BLEU']}  chrF: {results['chrF']}  METEOR: {results['METEOR']}")

    # --- 5. Output Results ---
    results_df = pd.DataFrame(all_results)
    print("\n" + "="*40)
    print("M2M100 Original PERFORMANCE:")
    print("="*40)
    print(results_df.to_string(index=False))

    results_df.to_csv("m2m100_benchmark_results.csv", index=False)
    print("\nResults saved to m2m100_or_benchmark_results.csv")
