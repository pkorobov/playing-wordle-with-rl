import string

GAME_VOCABULARY = ["sword", "crane", "plate"]
WORD_LENGTH = 5
MAX_TRIES = 6

LETTERS = ["<SOS>", "<SEP>", "<PAD>", *string.ascii_lowercase]
IDX2LETTER = LETTERS
LETTER2IDX = dict(zip(LETTERS, range(len(LETTERS))))

EMBEDDING_DIM = 64
