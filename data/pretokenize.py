import sys
import os
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import csv
import numpy as np

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tokenizer.bpe_tokenizer import BPETokenizerWrapper

def pretokenize_dataset(data_raw_dir: str, output_dir: str, tokenizer_path: str):
    data_raw_path = Path(data_raw_dir)
    output_path = Path(output_dir)

    # Resolve folder path conflict if a single file exists with folder name
    if output_path.is_file():
        output_path.unlink()
        
    output_path.mkdir(parents=True, exist_ok=True)

    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)

    for split in ["train", "val", "test"]:
        txt_file = data_raw_path / f"{split}.txt"
        csv_file = data_raw_path / f"{split}.csv"
        bin_file = output_path / f"{split}.bin"

        text = ""
        # 1. Check for .csv first, extract text content from columns
        if csv_file.exists():
            print(f"Reading CSV file: {csv_file}...")
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header row if present
                lines = []
                for row in reader:
                    if row:
                        lines.append(row[-1])  # Extract last or main text column
                text = "\n".join(lines)
        # 2. Fall back to .txt file
        elif txt_file.exists():
            print(f"Reading TXT file: {txt_file}...")
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue

        if not text:
            continue

        # 3. Tokenize text to ID sequence and save as uint16 binary
        encoded = tokenizer.encode(text)
        ids = np.array(encoded.ids, dtype=np.uint16)
        ids.tofile(bin_file)
        
        print(f"  --> Saved {len(ids):,} tokens to {bin_file} ({os.path.getsize(bin_file) / 1024:.1f} KB)")