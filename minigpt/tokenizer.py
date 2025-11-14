# tokenizer.py
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, s):  # string -> list[int]
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):  # list[int] -> string
        return ''.join(self.itos[i] for i in ids)
