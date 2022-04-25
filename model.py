import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical


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


class LSTMLayerCustom(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        # we want to use hidden states for attention, but in torch we need to get them manually
        # and thus we'll make two passes for bidirectional network
        self.num_layers = num_layers
        self.rnn_forward = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.rnn_backward = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, input):
        batch_size, max_seq_length = input.shape[0], input.shape[1]        

        hsl, csl = list(), list()
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        for i in range(max_seq_length):
            _, (h, c) = self.rnn_forward(input[:, i:i+1, :], (h, c))
            hsl.append(h)
            csl.append(c)
        hsl = torch.stack(hsl, dim=0) #.permute(0, 2, 1, 3).reshape(max_seq_length, batch_size, -1)
        csl = torch.stack(csl, dim=0) #.permute(0, 2, 1, 3).reshape(max_seq_length, batch_size, -1)

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hsr, csr = list(), list()
        for i in reversed(range(max_seq_length)):
            _, (h, c) = self.rnn_backward(input[:, i:i+1, :], (h, c))
            hsr.append(h)
            csr.append(c)

        hsr = list(reversed(hsr))
        csr = list(reversed(csr))
        hsr = torch.stack(hsr, dim=0) #.permute(0, 2, 1, 3).reshape(max_seq_length, batch_size, -1)
        csr = torch.stack(csr, dim=0) #.permute(0, 2, 1, 3).reshape(max_seq_length, batch_size, -1)

        hs, cs = torch.cat([hsl, hsr], dim=-1), torch.cat([csl, csr], dim=-1)

        return hs, cs


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.attention_map = None

    def forward(self, keys, queries, values, mask=None):
        """
        Input:
            keys: tensor of size batch_size x in_seq_length x input_size
            queries: tensor of size batch_size x out_seq_length x input_size (out_seq_length = 1 in our case)
            values: tensor of size batch_size x in_seq_length x input_size
        Output:
            output_value: tensor of size batch_size x input_size
        """

        attention_logits = torch.bmm(queries, keys.permute(dims=(0, 2, 1))) # batch_size x out_seq_length x in_seq_length
        # use masking here
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_logits = torch.where(mask, attention_logits, -torch.tensor(1e37))
        
        self.attention_map = torch.softmax(attention_logits, dim=-1) # batch_size x out_seq_length x in_seq_length
        outputs = torch.bmm(self.attention_map, values) # batch_size x out_seq_length x input_size

        return outputs


class Encoder(nn.Module):
    def __init__(self, letter_tokens, guess_tokens, emb_dim, hid_dim, num_layers, dropout, max_pos=6, pad_token=0):
        super().__init__()

        self.hid_dim = hid_dim

        self.max_pos = max_pos
        self.pos_embedding = nn.Embedding(max_pos, emb_dim)
        self.letter_embedding = nn.Embedding(letter_tokens, emb_dim)
        self.guess_state_embedding = nn.Embedding(guess_tokens, emb_dim)

        # self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, num_layers=2)
        pos_emb_size = 8
        self.rnn = LSTMLayerCustom(3 * emb_dim, hid_dim, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, letter_seq, state_seq):

        batch_size = letter_seq.shape[0]

        # letters_embedded = self.dropout(self.letter_embedding(letter_seq))
        # states_embedded = self.dropout(self.guess_state_embedding(state_seq))

        pos_embedded = self.pos_embedding(torch.arange(self.max_pos).reshape(1, -1)).repeat((batch_size, self.max_pos, 1))
        letters_embedded = self.letter_embedding(letter_seq)
        states_embedded = self.guess_state_embedding(state_seq)

        input_embeddings = torch.cat([letters_embedded, states_embedded, pos_embedded], dim=-1)

        # TODO: use pad token explicitly
        # lengths = (letter_seq != 0).sum(axis=-1)

        # input_embeddings = pack_padded_sequence(input_embeddings, lengths, batch_first=True, enforce_sorted=False)        
        hidden, cell = self.rnn(input_embeddings)
        
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # self.rnn = nn.LSTM(emb_dim, hid_dim, dropout=dropout, batch_first=True, num_layers=2)       
        self.rnn = nn.LSTM(emb_dim, hid_dim, dropout=dropout, batch_first=True, num_layers=num_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        # embedded = self.dropout(self.embedding(input))
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
                
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class RNNAgent(nn.Module):
    def __init__(self, letter_tokens, guess_tokens, emb_dim, hid_dim, output_dim,
                 game_voc_matrix, num_layers, output_len, sos_token, dropout=0.2):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.ouput_dim = output_dim
        self.num_layers = num_layers
        self.output_len = output_len

        # TODO: do not hardcode // 2
        self.encoder = Encoder(letter_tokens, guess_tokens, emb_dim, hid_dim // 2, num_layers, dropout)
        self.decoder = Decoder(output_dim, emb_dim, hid_dim, num_layers, dropout)
        self.attention = AttentionLayer(hid_dim)

        # TODO: do not hardcode * 2
        modules = [nn.Linear(hid_dim * num_layers, hid_dim), nn.ReLU(), nn.Linear(hid_dim, 1)]
        self.V_head = nn.Sequential(*modules)
        # TODO: do not hardcode * 2
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
        mask = (torch.arange(maxlen)[None, :] < lengths[:, None]).bool()

        # tensor to store decoder outputs
        batch_size = letter_seq.shape[0]
        logits = torch.zeros(batch_size, self.output_len + 1, self.letter_tokens)

        encoder_hiddens, encoder_cells = self.encoder(letter_seq, state_seq)        
        hidden, cell = encoder_hiddens.mean(dim=0), encoder_cells.mean(dim=0)
        # hidden, cell = encoder_hiddens.mean(dim=0)[:, -1:, :], encoder_cells.mean(dim=0)[:, -1:, :]
        # hidden, cell = hidden.permute(dims=(1, 0, 2)), cell.permute(dims=(1, 0, 2))
        encoder_hiddens = encoder_hiddens.permute(2, 0, 1, 3).reshape(batch_size, maxlen, -1)

        # compute V
        values = self.V_head(hidden.reshape(batch_size, -1))

        # first input to the decoder is the <sos> tokens
        input = torch.full(size=(batch_size,), fill_value=self.sos_token)
        
        letter_mask = torch.full(size=(batch_size, self.letter_tokens), fill_value=True)
        word_mask = torch.full(size=(batch_size, self.game_voc_matrix.shape[0]), fill_value=True)

        # logits: (seq_length, batch_size, num_classes)
        
        actions = torch.zeros(size=(batch_size, self.output_len), dtype=torch.long)
        log_probs = torch.zeros(size=(batch_size,))
        
        if self.debug_mode:
            fig, ax = plt.subplots(1, self.output_len)

        for t in range(1, self.output_len + 1):

            # cur_logits: (batch_size, num_classes)
            # actions: (batch_size,)
            _, hidden, cell = self.decoder(input, hidden, cell)
            
            decoder_hidden = hidden.permute(dims=(1, 0, 2)).reshape(batch_size, 1, -1)
            attentive_hidden = self.attention(encoder_hiddens, decoder_hidden, encoder_hiddens, mask).squeeze(1)
            
            if self.debug_mode:
                map_reshaped = self.attention.attention_map.squeeze().reshape(6, 6).detach().numpy()
                ax[t - 1].imshow(map_reshaped)
            
            cur_logits = self.logit_head(attentive_hidden)
            logits[:, t, :] = cur_logits
            probs = F.softmax(cur_logits, dim=-1)

            allowed_letters = get_allowed_letters(self.game_voc_matrix, word_mask, t-1)
            probs = torch.where(allowed_letters, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # torch.where(<your_tensor> != 0, <tensor with zeroz>, <tensor with the value>)
            actions_t = Categorical(probs=probs).sample()
            
            word_mask = word_mask & (self.game_voc_matrix[:, t - 1].unsqueeze(0) == actions_t.unsqueeze(1))

            # keep which words are acceptable
            cur_log_probs = torch.log(probs[range(batch_size), actions_t].clip(min=1e-12)).squeeze()

            # letters_allowed_count = allowed_letters.sum(axis=-1)
            # log_probs[letters_allowed_count > 1] += cur_log_probs[letters_allowed_count > 1]
            log_probs += cur_log_probs
            
            actions[:, t-1] = actions_t
            input = actions_t

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
        inputs = torch.LongTensor(inputs)
        letter_tokens, state_tokens = inputs[:, 0, :], inputs[:, 1, :]
        outputs = self(letter_tokens, state_tokens)
        return outputs