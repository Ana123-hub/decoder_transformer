import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def train_bpe_tokenizer(
    input_file: str = "data_raw/tinyshakespeare/train.csv",
    save_dir: str = ("artifacts/phase1_tinyshakespeare"),
    vocab_size: int = 1000  # Small vocab for TinyShakespeare; 8192/16384 for TinyStories
):
    """Trains a Byte-Level BPE tokenizer on raw text and saves tokenizer.json."""
    print(f"Training BPE Tokenizer on {input_file} (vocab_size={vocab_size})...")
    
    # Initialize Byte-Level BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Define trainer with special tokens
    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens
    )

    # Train on dataset
    tokenizer.train(files=[input_file], trainer=trainer)

    # Save trained tokenizer
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"Successfully saved tokenizer to {save_path}")

if __name__ == "__main__":
    train_bpe_tokenizer()