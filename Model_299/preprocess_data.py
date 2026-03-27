import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Configuration
DIALECTS = ['Chittagong', 'Sylhet', 'Barisal', 'Noakhali', 'Mymensingh']
VASHANTOR_BASE_PATH = './data/vashantor010'
OUTPUT_DIR = './data/processed/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


print("Step 1: Extracting Parallel Translation pairs from Vashantor...")
all_pairs = []

for dialect in DIALECTS:
    # Handle spelling variants in folder names if necessary
    folder_name = 'Barishal' if dialect == 'Barisal' else dialect

    for split in ['Train', 'Validation', 'Test']:
        file_path = f"{VASHANTOR_BASE_PATH}/{folder_name} {split} Translation.csv"
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, encoding='utf-8')
        # Clean column names because of trailing spaces found in your files
        df.columns = [c.strip().lower() for c in df.columns]

        # Identify standard and dialect columns
        # Standard is 'bangla_speech', Dialect is '[dialect]_bangla_speech'
        std_col = 'bangla_speech'
        dia_col = next(
            (c for c in df.columns if 'bangla_speech' in c and c != 'bangla_speech'), None)

        if std_col in df.columns and dia_col:
            for _, row in df.iterrows():
                all_pairs.append({
                    'text': clean_text(row[dia_col]),      # Input: Dialect
                    # Target: Standard Bengali
                    'standard': clean_text(row[std_col]),
                    'dialect_label': dialect               # For record keeping
                })

# Convert to DataFrame
full_df = pd.DataFrame(all_pairs).dropna()
full_df = full_df[full_df['text'] != ""]

# Split into Train/Val/Test (80/10/10)
train_df, temp_df = train_test_split(full_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save
train_df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_train.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_val.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_test.csv'), index=False)

print(f"Success! Prepared {len(full_df)} translation pairs.")
