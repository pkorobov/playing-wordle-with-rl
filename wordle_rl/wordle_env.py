import numpy as np
from typing import Optional
from termcolor import colored
import os
import gym
from gym import spaces

from wordle_rl.tokenizer import Tokenizer
from gym.utils import seeding


DEBUG_GAME_VOCABULARY = DEBUG_GAME_ANSWERS = ["sword", "crane", "plate"]
WORD_LENGTH = 5
MAX_TRIES = 6
EMBEDDING_DIM = 64


# state: 2 x 6 x 5 (guess, is_right)
class WordleEnv(gym.Env):
    def __init__(self, debug=False, reward_penalty_ratio=1.0):
        super().__init__()

        self.debug = debug
        self.reward_penalty_ratio = reward_penalty_ratio

        self.num_tries = 0
        # is_right matrix will contain 4 states per letter: empty, wrong, right, present in the word
        self.is_right = None
        # guess is the matrix of letter guesses during the current game
        self.guess = None
        self.word = None

        self.tokenizer = Tokenizer()
        self.game_voc_matrix = None
        self.game_ans_matrix = None
        self._initialize_vocabulary()

        self.action_space = spaces.MultiDiscrete([26] * 5)
        # I am not even sure that it is somehow used by gym or parallel wrapper
        self.observation_space = spaces.MultiDiscrete([26] * 2 * 5 * 6)
        self.reset()

    def _initialize_vocabulary(self):
        assert self.tokenizer is not None

        if not self.debug:
            with open(f'{os.path.dirname(__file__)}/../data/allowed_words.txt', 'r') as f:
                game_vocabulary = f.read().split()
        else:
            game_vocabulary = DEBUG_GAME_VOCABULARY
        
        self.game_voc_matrix = np.zeros(shape=(len(game_vocabulary), WORD_LENGTH), dtype=np.int32)
        for i in range(len(game_vocabulary)):
            for j, letter in enumerate(game_vocabulary[i]):
                self.game_voc_matrix[i, j] = self.tokenizer.letter2index[letter]

        if not self.debug:
            with open(f'{os.path.dirname(__file__)}/../data/possible_words.txt', 'r') as f:
                game_answers = f.read().split()
        else:
            game_answers = DEBUG_GAME_ANSWERS
        
        self.game_ans_matrix = np.zeros(shape=(len(game_answers), WORD_LENGTH), dtype=np.int32)
        for i in range(len(game_answers)):
            for j, letter in enumerate(game_answers[i]):
                self.game_ans_matrix[i, j] = self.tokenizer.letter2index[letter]

    def reset(self, seed: Optional[int] = None, **kwargs):

        super().reset(seed=seed, **kwargs)

        if self._np_random is not None:
            word_idx = self._np_random.randint(len(self.game_ans_matrix))
        else:
            word_idx = np.random.randint(len(self.game_ans_matrix))

        self.word = self.game_ans_matrix[word_idx]

        self.num_tries = 0

        self.is_right = np.zeros((MAX_TRIES, WORD_LENGTH), dtype=np.int32)
        self.guess = np.zeros((MAX_TRIES, WORD_LENGTH), dtype=np.int32)
        self.is_right[:, :] = self.tokenizer.guess_state2index['<PAD>']
        self.guess[:, :] = self.tokenizer.letter2index['<PAD>']

        obs = np.stack([self.guess, self.is_right])
        return obs

    def compute_pattern(self, action: np.ndarray):
        equality_grid = np.equal.outer(action, self.word)
        pattern = np.full((WORD_LENGTH,), fill_value=self.tokenizer.guess_state2index['<MISS>'])

        # Green pass
        green_matches = np.diag(equality_grid).copy()
        pattern[green_matches] = self.tokenizer.guess_state2index['<RIGHT>']  # right
        equality_grid[green_matches, :] = False
        equality_grid[:, green_matches] = False

        # Yellow pass
        idx = np.argwhere(equality_grid == True)
        for i, j in idx:
            if equality_grid[i, j]:
                pattern[i] = self.tokenizer.guess_state2index['<CONTAINED>']  # MISPLACED
                equality_grid[:, j] = False
                equality_grid[i, :] = False
        return pattern, green_matches

    def step(self, action: np.ndarray):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info: None TODO: use it somehow
        """
        
        action = action.squeeze()
        assert len(action.shape) == 1, action.shape
        assert len(action) == WORD_LENGTH, len(action)

        info = dict()

        self.is_right[self.num_tries, :], green_matches = self.compute_pattern(action)
        self.guess[self.num_tries, :] = action

        reward, done = green_matches.sum() / WORD_LENGTH, False
        reward *= self.reward_penalty_ratio

        if self.num_tries > 0:
            reward -= (
                (self.guess[:self.num_tries] == action) & 
                (self.is_right[:self.num_tries] == self.tokenizer.guess_state2index['<MISS>'])
            ).any(axis=0).sum() / WORD_LENGTH

        if green_matches.all() or self.num_tries == MAX_TRIES - 1:
            if green_matches.all():
                reward = 10.0
            done = True
            obs = self.reset()
        else:
            self.num_tries += 1
            obs = np.stack([self.guess, self.is_right])

        return obs, reward, done, info

    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        for i in range(self.guess.shape[0]):
            s = []
            for j in range(self.guess.shape[1]):
                letter = " "
                if self.guess[i, j] != 0:
                    letter = self.tokenizer.index2letter[self.guess[i, j]]
                c = None
                if self.is_right[i, j] == self.tokenizer.guess_state2index['<RIGHT>']:
                    c = "green"
                elif self.is_right[i, j] == self.tokenizer.guess_state2index['<CONTAINED>']:
                    c = "yellow"
                s.append(colored(letter, color=c))
            print("".join(s), end="\n")


if __name__ == "__main__":
    import time
    from wordle_rl.tokenizer import Tokenizer

    env = WordleEnv()
    tokenizer = Tokenizer()
    # env.word = np.array([tokenizer.letter2index[c] for c in "zesty"])
    print([tokenizer.index2letter[c] for c in env.word])

    while True:
        # you can play here with env on your own
        a = input()
        a = np.array([tokenizer.letter2index[c] for c in a])
        obs, reward, done, info = env.step(a.reshape(1, -1))
        env.render()
        time.sleep(0.5)
