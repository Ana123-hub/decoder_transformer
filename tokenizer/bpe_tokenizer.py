import os
from typing import List, Union
from tokenizers import Tokenizer

class BPETokenizerWrapper:
    """Wrapper class for loading and running inference with our trained BPE tokenizer."""
    
    def __init__(self, tokenizer_path: str = "artifacts/phase1_tinyshakespeare/tokenizer.json"):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}. Run train_tokenizer.py first.")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        """Encodes string into a list of integer token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: Union[List[int], List[List[int]]]) -> str:
        """Decodes token ID list back into string text."""
        if len(ids) > 0 and isinstance(ids[0], list):
            return self.tokenizer.decode_batch(ids)
        return self.tokenizer.decode(ids)