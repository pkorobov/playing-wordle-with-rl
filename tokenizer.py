import string

common_tokens = ["<PAD>", "<SOS>", "<EOS>"]
guess_states = common_tokens + ["<RIGHT>", "<CONTAINED>", "<MISS>"]


class Tokenizer:
    def __init__(self):
        common_tokens = ["<PAD>", "<SOS>", "<EOS>"]
        self.index2letter = common_tokens[:] + list(string.ascii_lowercase)
        self.letter2index = dict(zip(self.index2letter, range(len(self.index2letter))))

        self.index2guess_state = common_tokens[:] + ["<RIGHT>", "<CONTAINED>", "<MISS>"]
        self.guess_state2index = dict(zip(self.index2guess_state, range(len(self.index2guess_state))))
