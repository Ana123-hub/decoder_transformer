import os
import sys
import urllib.request
from pathlib import Path

# Add project root directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tokenizer.train_tokenizer import train_bpe_tokenizer
from data.pretokenize import pretokenize_dataset

TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def prepare_tinyshakespeare():
    raw_dir = Path("data_raw/tinyshakespeare")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = raw_dir / "input.txt"
    train_file = raw_dir / "train.csv"
    val_file = raw_dir / "val.csv"

    # 1. Download raw input text if not present
    if not input_file.exists():
        print(f"Downloading TinyShakespeare from {TINYSHAKESPEARE_URL}...")
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, input_file)
        print(f"Saved to {input_file}")

    # 2. Split raw text into train (90%) and validation (10%) sets
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()

    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    with open(train_file, "w", encoding="utf-8") as f:
        f.write(train_data)
    with open(val_file, "w", encoding="utf-8") as f:
        f.write(val_data)

    print(f"Data split: Train = {len(train_data):,} chars | Val = {len(val_data):,} chars")

    # 3. Train Byte-Pair Encoding (BPE) tokenizer
    print("\n--- Training Tokenizer ---")
    train_bpe_tokenizer(
        input_file=str(train_file),
        save_dir="tokenizer",
        vocab_size=1000  # 1,000 token vocabulary for TinyShakespeare
    )

    # 4. Pre-tokenize text into train.bin and val.bin
    print("\n--- Encoding Raw Text to Binary Format ---")
    pretokenize_dataset(
        data_raw_dir="data_raw/tinyshakespeare",
        output_dir="data/tinyshakespeare",
        tokenizer_path="tokenizer/tokenizer.json"
    )
    print("\nTinyShakespeare dataset preparation complete!")

if __name__ == "__main__":
    prepare_tinyshakespeare()