import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from a2c import DEVICE
num_letters = 29


def get_allowed_letters(word_matrix, word_mask, position):
    """
        word_matrix: torch.Tensor of size (num_words, word_length)
        word_mask: torch.Tensor of size (batch_size, num_words)
        position: int
        
        returns
        
        letters_mask: (batch_size, num_letters) -- mask of possible letters
    """
    batch_size = word_mask.size(0)
    word_matrix_expanded = word_matrix[:, position].unsqueeze(0).expand(batch_size, -1)
    
    # print(word_matrix[:, position].shape)
    # print(word_matrix_expanded.shape)
    # print(word_matrix[:, position].unsqueeze(1).shape)
    
    word_matrix_masked = (word_matrix_expanded * word_mask).long()    
    letter_mask = torch.full(fill_value=False, size=(batch_size, num_letters))

    # letter_mask = letter_mask.scatter(index=word_matrix_masked, dim=1, value=True)
    rows = torch.arange(0, letter_mask.size(0))[:, None]
    n_col = word_matrix_masked.size(1)
    letter_mask[rows.repeat(1, n_col), word_matrix_masked] = 1

    letter_mask[:, 0] = 0
    return letter_mask


class Encoder(nn.Module):
    def __init__(self, letter_tokens, guess_tokens, emb_dim, hid_dim, num_layers, dropout, max_pos=6, pad_token=0):
        super().__init__()

        self.hid_dim = hid_dim

        self.max_pos = max_pos
        self.pos_embedding = nn.Embedding(max_pos, emb_dim)
        self.letter_embedding = nn.Embedding(letter_tokens, emb_dim)
        self.guess_state_embedding = nn.Embedding(guess_tokens, emb_dim)

        self.rnn = nn.LSTM(3 * emb_dim, hid_dim, num_layers, bidirectional=True, batch_first=True)

        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, letter_seq, state_seq):

        batch_size = letter_seq.shape[0]

        # letters_embedded = self.dropout(self.letter_embedding(letter_seq))
        # states_embedded = self.dropout(self.guess_state_embedding(state_seq))

        pos_embedded = self.pos_embedding(
                            torch.arange(self.max_pos).reshape(1, -1).to(DEVICE)
                        ).repeat((batch_size, self.max_pos, 1))
        letters_embedded = self.letter_embedding(letter_seq)
        states_embedded = self.guess_state_embedding(state_seq)

        input_embeddings = torch.cat([letters_embedded, states_embedded, pos_embedded], dim=-1)

        # embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(input_embeddings)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers, batch_first=True)

        # self.energy = nn.Linear(hidden_size * 3, 1)
        self.energy = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.fc = nn.Linear(hidden_size * 3, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.attention = None

    def forward(self, x, encoder_states, hidden, cell, attention_mask=None):
        x = x.unsqueeze(1)
        # x: (N, 1) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (N, 1, embedding_size)

        sequence_length = encoder_states.shape[1]
        h_reshaped = hidden.repeat(sequence_length, 1, 1).permute(1, 0, 2)
        # h_reshaped: (N, seq_length, hidden_size*2)

        energy = self.energy(torch.cat((h_reshaped, encoder_states), dim=2))
        # energy: (N, seq_length, 1)

        if attention_mask is not None:
            # TODO: возможно: стоит убрать sos токены из attention
            # attention_mask: (N, seq_length)
            attention_mask = attention_mask.unsqueeze(-1)
            energy = torch.where(attention_mask, energy, -torch.tensor(1e37).to(DEVICE))

        self.attention = self.softmax(energy)
        # attention: (N, seq_length, 1)

        # attention: (N, seq_length, 1), snk
        # encoder_states: (N, seq_length, hidden_size*2), snl
        # we want context_vector: (N, 1, hidden_size*2), i.e knl
        context_vector = torch.einsum("nsk,nsl->nkl", self.attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (N, 1, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (N, 1, hidden_size)

        fc_input = torch.cat([outputs, context_vector], dim=-1)
        # outputs shape: (N, 1, hidden_size*3)

        predictions = self.fc(fc_input).squeeze(1)
        # predictions: (N, output_size)

        return predictions, hidden, cell


class RNNAgent(nn.Module):
    def __init__(self, letter_tokens, guess_tokens, emb_dim, hid_dim, output_dim,
                 game_voc_matrix, num_layers, output_len=5, sos_token=1, dropout=0.2):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.ouput_dim = output_dim
        self.num_layers = num_layers
        self.output_len = output_len

        self.encoder = Encoder(letter_tokens, guess_tokens, emb_dim, hid_dim, num_layers, dropout)
        self.decoder = Decoder(letter_tokens, emb_dim, hid_dim, output_dim, num_layers, dropout)

        modules = [nn.Linear(hid_dim * num_layers, hid_dim), nn.ReLU(), nn.Linear(hid_dim, 1)]
        self.V_head = nn.Sequential(*modules)
        self.logit_head = nn.Linear(hid_dim * num_layers, output_dim)
        
        self.letter_tokens = letter_tokens
        self.game_voc_matrix = game_voc_matrix
        self.sos_token = sos_token
        self.debug_mode = False

    def debug(self, mode=False):
        self.debug_mode = mode
    
    def forward(self, letter_seq, state_seq):
        """
            inputs:
                letter_seq: (batch_size x sequence_length)
                state_seq: (batch_size x sequence_length)

            outputs:
                
        """
        
        maxlen = letter_seq.shape[1]
        lengths = (letter_seq != 0).sum(axis=-1)
        attention_mask = (torch.arange(maxlen)[None, :].to(DEVICE) < lengths[:, None]).bool()

        # tensor to store decoder outputs
        batch_size = letter_seq.shape[0]
        logits = torch.zeros(batch_size, self.output_len, self.letter_tokens)

        encoder_states, hidden, cell = self.encoder(letter_seq, state_seq)

        # compute V
        values = self.V_head(hidden.reshape(batch_size, -1))

        # first input to the decoder is the <sos> tokens
        x = torch.full(size=(batch_size,), fill_value=self.sos_token).to(DEVICE)
        
        word_mask = torch.full(size=(batch_size, self.game_voc_matrix.shape[0]), fill_value=True)

        # logits: (seq_length, batch_size, num_classes)
        
        actions = torch.zeros(size=(batch_size, self.output_len), dtype=torch.long, device=DEVICE)
        log_probs = torch.zeros(size=(batch_size,), device=DEVICE)
        
        if self.debug_mode:
            fig, ax = plt.subplots(1, self.output_len)

        for t in range(self.output_len):

            # cur_logits: (batch_size, num_classes)
            # actions: (batch_size,)
            cur_logits, hidden, cell = self.decoder(x, encoder_states, hidden, cell, attention_mask)

            if self.debug_mode:
                map_reshaped = self.decoder.attention.squeeze().reshape(6, 6).cpu().detach().numpy()
                ax[t].imshow(map_reshaped)
            
            logits[:, t, :] = cur_logits
            probs = F.softmax(cur_logits, dim=-1)

            allowed_letters = get_allowed_letters(self.game_voc_matrix, word_mask, t).to(DEVICE)
            probs = torch.where(allowed_letters, probs, torch.zeros_like(probs).to(DEVICE))
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # torch.where(<your_tensor> != 0, <tensor with zeroz>, <tensor with the value>)
            actions_t = Categorical(probs=probs).sample()
            
            word_mask = word_mask & (self.game_voc_matrix[:, t].unsqueeze(0) == actions_t.unsqueeze(1).cpu())

            # keep which words are acceptable
            cur_log_probs = torch.log(probs[torch.arange(batch_size), actions_t].clip(min=1e-12)).squeeze()

            log_probs += cur_log_probs
            
            actions[:, t] = actions_t
            x = actions_t

        if self.debug_mode:
            plt.show()

        return {
            "actions": actions.cpu().numpy(),
            # "logits": logits,
            "log_probs": log_probs,
            "values": values.reshape(-1),
        }
    
    def act(self, inputs):
        '''
        input:
            inputs - numpy array, (batch_size x sequences x sequence_length)
        output: dict containing keys ['actions', 'logits', 'log_probs', 'values']:
            'actions' - selected actions, numpy, (batch_size, sequence_length)
            'log_probs' - log probs of selected actions, tensor, (batch_size)
            'values' - critic estimations, tensor, (batch_size)
        '''
        inputs = torch.LongTensor(inputs).to(DEVICE)
        letter_tokens, state_tokens = inputs[:, 0, :], inputs[:, 1, :]
        outputs = self(letter_tokens, state_tokens)
        return outputs