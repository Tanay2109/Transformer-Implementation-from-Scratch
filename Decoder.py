import torch.nn as nn
import torch
from Transformer import SelfAttention, TransformerBlock
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention=SelfAttention(embed_size, heads)
        self.norm= nn.LayerNorm(embed_size)
        self.transformer_block= TransformerBlock( embed_size, heads, dropout, forward_expansion)

        self.dropout= nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention= self.attention(x,x,x, trg_mask)
        query= self.dropout(self.norm(attention + x))
        out= self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length,):
        super(Decoder,self).__init__()
        self.device= device
        self.word_embedding= nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding= nn.Embedding(max_length, embed_size)

        self.layers= nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range (num_layers)]
        )
        self.fc_out=nn.Linear(embed_size, trg_vocab_size)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len= x.shape
        positions= torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x= self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x=layer(x, enc_out, enc_out, src_mask, trg_mask)

        out= self.fc_out(x)
        return out

