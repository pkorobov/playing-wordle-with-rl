import string


class LetterTokenizer:
    def __init__(self):
        self.index2letter = ["<PAD>", "<SOS>", "<EOS>"]
        self.letter2index = dict(zip(self.index2letter, range(len(self.index2letter))))

        for i, char in enumerate(string.ascii_lowercase):
            self.index2letter.append(char)
            self.letter2index[char] = self.n_tokens()

    def n_tokens(self):
        return len(self.index2letter)
