import math

import torch
import torch.nn as nn

from modules.transformer_block import TransformerBlock

class PatchEmbedder(nn.Module):
    """ 
    Takes a batch of image tensors and splits them each into N patches, and then
    embeds them into N, E-dimensional vectors / tokens
    """
    
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):  
        """
        By default, as in the DINO paper, we use RGB images with a desired patch size of 16, and 
        embedding dimension of 768
        """
        
        super().__init__()
        
        # This is not *really* a convolutional layer - it acts like an MLP but clever use
        # of the stride length here makes it easier embed non-overlapping patches 
        self.embedder = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """ 
        A forward pass in this case takes an image with shape (B, C, W, H) and returns a batch
        which each has N patch embeddings with dimension E
        """
        
        # From (B, C, W, H) to (B, E, sqrt(N), sqrt(N))
        x = self.embedder(x)
        # to (B, E, N)
        x = x.flatten(2)
        # to (B, N, E)
        x = x.transpose(1,2)
        
        return x

class VisionTransformer(nn.Module):
    """
    The Vision Transformer module is essentially just a stack of transformer blocks, with 
    patch embedding and class token prepention at the front. It follows the implementation from 
    Dosovitskiy et al. with modifications as per the DINO paper
    """
    
    def __init__(
        self, 
        global_size=224,
        n_classes=32,
        in_channels=3,
        patch_size=16,
        embed_dim=768,
        n_blocks=12,
        n_heads=12,
        qkv_drop_p=0., 
        embed_drop_p=0.,
        mlp_hidden_ratio=4.0,
        mlp_drop_p=0.
    ):
        """
        The inputs here allow us to instantiate the entire network
        """
        super().__init__()
        
        # Start by creating the patch embedding
        self.patch_embedder = PatchEmbedder(in_channels=in_channels, 
                                            patch_size=patch_size, 
                                            embed_dim=embed_dim)
        
        # We need to prepend a class token - which is an E-dimensional set of learnable parameters
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Similarly we need the position embeddings - these are a set of T, E-dimensional learnable
        # vectors. We set them up for the global size crops and interpolate to the local
        # size crops as needed
        self.patch_size = patch_size
        self.n_global_patches = (global_size // self.patch_size)**2
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.n_global_patches + 1, embed_dim))
        
        # Then we have a block of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(io_dim=embed_dim, n_heads=n_heads, 
                             qkv_drop_p=qkv_drop_p, embed_drop_p=embed_drop_p,
                             mlp_hidden_ratio=mlp_hidden_ratio, mlp_drop_p=mlp_drop_p)
            for i in range(n_blocks)
        ])
        
        # A final normalisation layer
        self.norm = nn.LayerNorm(embed_dim)
        
        # A classification head is not actually needed here. We can just softmax the CLS token
        # outputs (each an E-dimensional vector) and use those to calculate the probability
        # distributions as per the DINO paper's equation 1. A use case is to take these CLS
        # tokens and use an MLP head to classify them with labelled data (fine tuning)
        
    def get_pos_encoding(self, x, W, H):
        """ 
        The positional encoding is a learned tensor of size (1, T, E). T is variable due to 
        local and global views. We defaultly build for the global view, so for local
        views we need to perform bicubic interpolation. The input tensors here can be of 
        size (B, T, E), where T is variable
        """
        
        # N = T - 1
        this_N = x.shape[1] - 1
        glob_N = self.n_global_patches
        E = x.shape[-1]
        
        # If there are the number of patches that we expect in a global patch then
        # the position encoding requires no interpolation
        if this_N == glob_N:
            return self.pos_encoding
        
        # If not then we need to rescale
        w_scale = W // self.patch_size # e.g., 96 / 16 = 6 (local size / patch size)
        h_scale = H // self.patch_size
        
        # Add a small number to avoid floating point error that can occur during interpolation
        w_scale += 0.1
        h_scale += 0.1
        
        # Separate the CLS token position embedding from the remaining
        cls_encoding = self.pos_encoding[:,0]
        rem_encoding = self.pos_encoding[:,1:]
        
        # Perform the recommended bicubic interpolation
        sqrt_gN = int(math.sqrt(glob_N)) # In the standard parameterisation this will be 
                                         # 14 = sqrt(196)
        interp_rem_encoding = nn.functional.interpolate(
            rem_encoding.reshape(1, sqrt_gN, sqrt_gN, E).permute(0, 3, 1, 2), # to (1, E, sqN, sqN)
            scale_factor=(w_scale / sqrt_gN, h_scale / sqrt_gN),
            mode='bicubic'
        )
        
        # Get back into form (1, T, E) and concat with unmodified class token position encoding
        interp_rem_encoding = interp_rem_encoding.permute(0, 2, 3, 1).view(1, -1, E)
        interp_pos_encoding = torch.cat((cls_encoding.unsqueeze(0), interp_rem_encoding), dim=1)
        return interp_pos_encoding
        
    def forward(self, x):
        """
        Perform the forward pass and return either logits or embeddings. Note that we do not expect
        jagged tensors here: i.e., a single batch can't have BOTH global and local views in it, only
        entirely global or entirely local views
        """
        
        B, C, W, H = x.shape
        
        # Begin by setting up the tokens by prepending the class token to each batch item's patch
        # embeddings, then add on the position encodings (interpolated as necessary)
        # to (B, N, E)
        x = self.patch_embedder(x)
        # Remember that each element of the batch needs one of these
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # to (B, T, E)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.get_pos_encoding(x, W, H)
        
        # TODO dropout on positional encodings
        
        # Then its just a matter of running it through all the blocks
        for block in self.blocks:
            x = block(x)
        
        # And the normalisation
        x = self.norm(x)
        
        # Then we just return the CLS tokens (first token of every batch - each should be 
        # E-dimensional)
        return x[:, 0]