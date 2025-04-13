import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size= embed_size
        self.heads= heads
        self.head_dims= embed_size//heads

        assert (self.head_dims*heads == embed_size), "Embed size needs to be div by heads."

        self.values= nn.Linear(self.head_dims, self.head_dims, bias= False)
        self.keys= nn.Linear(self.head_dims, self.head_dims, bias= False)
        self.queries= nn.Linear(self.head_dims, self.head_dims, bias= False)
        self.fc_out= nn.Linear(heads*self.head_dims, embed_size)

    def forward(self, values, keys, query, mask):
        N= query.shape[0]
        value_len, key_len, query_len= values.shape[1], keys.shape[1], query.shape[1]

        #Split embedding into self.heads pieces
        values= values.reshape(N, value_len, self.heads, self.head_dims)
        keys= keys.reshape(N, key_len, self.heads, self.head_dims)
        queries= queries.reshape(N, key_len, self.heads, self.head_dims)

        energy= torch.einsum("nqhd,nkhd->nhqk", [queries,keys])

        if mask is not None:
            energy= energy.masked_fill(mask==0, float("-1e20"))

        attention= torch.softmax(energy/(self.embed_size**(1/2)), dim=3)

        out=torch.einsum("nhql,nlhd->nqhd", [attention,values]).reshape(N, query_len, self.heads.head_dims)

        out= self.fc_out(out)
        return out
