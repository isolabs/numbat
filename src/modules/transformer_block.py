import torch.nn as nn

class MultiheadAttention(nn.Module):
    """
    This module performs the self attention mechanism
    """
    def __init__(self, io_dim, n_heads=12, qkv_drop_p=0., embed_drop_p=0.):
        super().__init__()

        # Multi head attention differs from single head attention by having multiple
        # heads which each take in a fraction of the input vector, processing the
        # whole thing in parllel
        self.n_heads  = n_heads
        self.io_dim   = io_dim
        self.head_dim = self.io_dim // self.n_heads

        # This prevents feeding large values into the softmax (causing small gradients)
        self.scale = self.head_dim ** -0.5

        # We need an mapping which takes in the input vector and generates a query, key, and
        # value vector. Its faster to do this all together, rather than with 3 separate operations
        self.qkv_extractor = nn.Linear(self.io_dim, self.io_dim * 3, bias=True)
        self.qkv_drop      = nn.Dropout(qkv_drop_p)
        # This linear mapping takes the concatenated head outputs and creates a representation
        self.embedder      = nn.Linear(self.io_dim, self.io_dim)
        self.embedder_drop = nn.Dropout(embed_drop_p)

    def forward(self, x):
        """
        Takes a vector of shape (B, N+1, E), extracts the query, key and value, across multiple
        heads, applies the qkv operation, softmaxes, and outputs a vector of (B, N+1, E)
        """

        # Input shape. Note that T, the number of tokens, will be N (the number of patches) + 1, due
        # to the CLS token
        B, T, E = x.shape

        # to (B, T, 3E)
        qkv = self.qkv_extractor(x)

        # Reshape to (B, T, 3, NH, HD)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        # Permute the axes to (3, B, NH, T, HD),
        # so that it can be processed in parallel by many 'heads'
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Apply the 'selection' operation
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Firstly, dot the keys and queries, then softmax the result so that its a probability dist,
        # note that we have to swap the HD and T axes to allow for a matrix multiplication. The
        # dimension is (B, NH, T, T)
        dp = (q @ k.transpose(-2, -1)) * self.scale
        dp_pd = dp.softmax(dim=-1)
        dp_pd = self.qkv_drop(dp_pd)

        # Secondly, multiply this result with the values, to get a scaled selection
        # in the shape (B, NH, T, HD)
        scaled_sel = dp_pd @ v
        # and then the shape (B, T, NH, HD)
        scaled_sel = scaled_sel.transpose(1,2)

        # Ensure that this has the same dimensions as the input (B, T, E),
        # since NH * HD = E by design
        scaled_sel = scaled_sel.flatten(2)

        # Now we just have to run this through an embedding layer, which keeps the shape the same
        output = self.embedder(scaled_sel)
        output = self.embedder_drop(output)

        return output

class MultiLayerPerceptron(nn.Module):
    """
    This module represents the MLP which is used in a transformer block (towards the end)
    """
    def __init__(self, io_dim, hidden_ratio, drop_p=0.):
        """
        As in the Dosovitskiy et al. paper we use GELU activations and dropout to prevent
        overfitting
        """
        super().__init__()
        self.n_feat_hidden = int(io_dim * hidden_ratio)
        self.fc1  = nn.Linear(io_dim, self.n_feat_hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(self.n_feat_hidden, io_dim)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        """
        Performs a forward pass of this MLP. Shape goes from (B, T, E) to (B, T, E)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    A single transformer encoder / block. Follows the implementation described in Dosovitskiy et al.
    where we perform layer normalisation, multihead attention, layer normalisation, then MLP
    embedding, all the while using residual skip connections
    """

    def __init__(self, io_dim, n_heads=12, qkv_drop_p=0., embed_drop_p=0.,
                 mlp_hidden_ratio=4.0, mlp_drop_p=0.):
        super().__init__()

        # The following computations are performed in order

        # Note that we need two different layer normalisation layers because they have
        # learnable parameters
        self.norm1  = nn.LayerNorm(io_dim, eps=1e-6) # This epsilon matches the paper details

        self.mhattn = MultiheadAttention(io_dim=io_dim, n_heads=n_heads,
                                         qkv_drop_p=qkv_drop_p, embed_drop_p=embed_drop_p)

        self.norm2  = nn.LayerNorm(io_dim, eps=1e-6) # This epsilon matches the paper details

        self.mlp = MultiLayerPerceptron(io_dim=io_dim,
                                        hidden_ratio=mlp_hidden_ratio,
                                        drop_p=mlp_drop_p)

    def forward(self, x):
        """
        Perform the forward pass operation, using residual links to propagate information. The
        shape again is preserved (B, T, E)
        """
        x = x + self.mhattn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x