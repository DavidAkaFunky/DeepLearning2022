import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
    ):

        super(Attention, self).__init__()
        "Luong et al. general attention (https://arxiv.org/pdf/1508.04025.pdf)"
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        query,
        encoder_outputs,
        src_lengths,
    ):
        # query: (batch_size, max_tgt_len, hidden_dim)
        # encoder_outputs: (batch_size, max_src_len, hidden_dim) -> context
        # src_lengths: (batch_size)
        src_seq_mask = ~self.sequence_mask(src_lengths)
        z = self.linear_in(query)
        attn_scores = torch.bmm(z, encoder_outputs.transpose(1, 2))
        attn_scores = attn_scores.masked_fill_(src_seq_mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.softmax(attn_scores, 2)
        c = torch.bmm(attn_weights, encoder_outputs)
        attn_out = torch.tanh(self.linear_out((torch.cat([query, c], dim=2))))
        # attn_out: (batch_size, max_tgt_len, hidden_size)
        return attn_out

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (
            torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        emb = self.dropout(self.embedding(src))
        emb = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        output, final_hidden = self.lstm(emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        enc_output = self.dropout(output)
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        return enc_output, final_hidden


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)
        tgt = tgt[:,:-1] if tgt.shape[1] > 1 else tgt # tgt[:,:-1] or tgt[:,1:]
        emb = self.dropout(self.embedding(tgt))
        outputs, dec_state = self.lstm(emb, dec_state)
        if self.attn is not None:
            outputs = self.attn(
                outputs,
                encoder_outputs,
                src_lengths,
            )
        outputs = self.dropout(outputs)
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)
        return outputs, dec_state


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state
            
        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )
        
        return self.generator(output), dec_hidden
