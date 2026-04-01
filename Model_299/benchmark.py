import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, M2M100Tokenizer
import evaluate
from tqdm import tqdm

# --- 1. Setup and Load Metrics ---
print("Loading evaluation metrics...")
sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
chrf = evaluate.load("chrf")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 2. Load Models Exactly as in main.py ---
print("Loading models...")
t5_model_name = "csebuetnlp/banglat5"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name, use_fast=False)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(device)

m2m100_model_name = "facebook/m2m100_418M"
m2m_tokenizer = M2M100Tokenizer.from_pretrained(m2m100_model_name)
m2m_model = M2M100ForConditionalGeneration.from_pretrained(
    m2m100_model_name).to(device)

# --- 3. Define Generation Functions (Replicated from main.py) ---


def generate_t5(input_text):
    text = "paraphrase: " + input_text
    input_ids = t5_tokenizer(text, return_tensors="pt").input_ids.to(device)
    outputs = t5_model.generate(
        input_ids, max_length=64, num_beams=4, no_repeat_ngram_size=2,
        repetition_penalty=3.5, length_penalty=1.0, early_stopping=True
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_m2m(input_text):
    m2m_tokenizer.src_lang = "bn"
    encoded_bn = m2m_tokenizer(input_text, return_tensors="pt").to(device)
    generated_tokens = m2m_model.generate(
        **encoded_bn,
        forced_bos_token_id=m2m_tokenizer.get_lang_id("bn"),
        max_length=64
    )
    return m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# --- 4. Evaluation Logic ---


def run_evaluation(source_texts, target_texts, model_name):
    predictions = []

    for text in tqdm(source_texts, desc=f"Evaluating {model_name}"):
        if model_name == "BanglaT5":
            pred = generate_t5(text)
        else:
            pred = generate_m2m(text)
        predictions.append(pred)

    # Calculate Metrics
    bleu_result = sacrebleu.compute(predictions=predictions, references=[
                                    [t] for t in target_texts])
    rouge_result = rouge.compute(
        predictions=predictions, references=target_texts)
    chrf_result = chrf.compute(predictions=predictions, references=[
                               [t] for t in target_texts])

    return {
        "Model": model_name,
        "BLEU": round(bleu_result["score"], 2),
        # Convert to percentage
        "ROUGE-L": round(rouge_result["rougeL"] * 100, 2),
        "chrF": round(chrf_result["score"], 2)
    }


# --- 5. Load Data and Run ---
if __name__ == "__main__":
    # Load data
    ctg_df = pd.read_csv("Chittagong Test Translation.csv")
    syl_df = pd.read_csv("Sylhet Test Translation.csv")

    # Strip column names just in case
    ctg_df.columns = ctg_df.columns.str.strip()
    syl_df.columns = syl_df.columns.str.strip()

    datasets = {
        "Chittagong": (ctg_df['chittagong_bangla_speech'].dropna().tolist(), ctg_df['bangla_speech'].dropna().tolist()),
        "Sylhet": (syl_df['sylhet_bangla_speech'].dropna().tolist(), syl_df['bangla_speech'].dropna().tolist())
    }

    models = ["BanglaT5", "M2M100"]
    all_results = []

    for region, (source, target) in datasets.items():
        print(f"\n--- Benchmarking {region} Dialect ---")
        for model in models:
            results = run_evaluation(source, target, model)
            results["Dialect"] = region
            all_results.append(results)

    # --- 6. Display Results ---
    results_df = pd.DataFrame(all_results)
    # Reorder columns for readability
    results_df = results_df[["Dialect", "Model", "BLEU", "ROUGE-L", "chrF"]]

    print("\n=== Final Benchmark Results ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to 'benchmark_results.csv'")
