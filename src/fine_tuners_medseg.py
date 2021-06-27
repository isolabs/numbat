
import math

import torch
import torch.nn as nn
from torchvision import transforms

class FineTunerSegmentation(nn.Module):
    """
    The fine tuner fundamentally works by wrapping a vision transformer and using its
    outputs to perform segmentation with some further simple layers
    """
    
    def __init__(
        self, 
        vit, 
        vit_embed_dim,
        patch_size,
        n_heads_classes,
        label_map_size,
    ):
        """
        Instantiate the 'extra' classification layers
        """

        super().__init__()

        # Save the VIT
        self.vit = nn.ModuleList([vit]) # Register as a trainable module
        self.vit_embed_dim = vit_embed_dim
        self.patch_size = patch_size
        self.n_heads_classes = n_heads_classes

        #print(f"Learnable parameters in VIT: {sum(p.numel() for p in self.vit[0].parameters() if p.requires_grad)}")

        # Just need to take the VIT output attn layers, reshape them to 
        # channels=head attentions x W x H and then resize + argmax to get the 
        # segmentation output
        self.label_map_size = label_map_size
        # 32x32 to 64x64
        self.up1 = nn.ConvTranspose2d(in_channels=n_heads_classes, out_channels=n_heads_classes, kernel_size=7, stride=2, padding=3, output_padding=1)
        # 64x64 to 128x128
        self.up2 = nn.ConvTranspose2d(in_channels=n_heads_classes, out_channels=n_heads_classes, kernel_size=7, stride=2, padding=3, output_padding=1)
        # 128x128 to 256x256
        self.up3 = nn.ConvTranspose2d(in_channels=n_heads_classes, out_channels=n_heads_classes, kernel_size=7, stride=2, padding=3, output_padding=1)

        # Seg class predictions (channel/class wise)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Through the VIT and then through the classification layers
        """
        
        B, C, W, H = x.shape
    
        # Shape is currently (B, C, W, H)
        image = x
        new_w = image.shape[2] - image.shape[2] % self.patch_size
        new_h = image.shape[3] - image.shape[3] % self.patch_size
        image = image[:, :, :new_w, :new_h]

        # The output map will be this big
        w_map = image.shape[2] // self.patch_size
        h_map = image.shape[3] // self.patch_size

        # Get the output, shape is (B, n_heads, X=Y^2 + 1, X=Y^2 + 1),
        # where Y is the W/H
        attn = self.vit[0](image, return_only_last_attn=True)
        n_heads = attn.shape[1]

        # We only want the CLS token SELF attention, so not the heads
        # attention on itself in the last dim, and only the first dim on
        # the second to last dim
        attn = attn[:, :, 0, 1:]                        # (B, n_heads, 1, X - 1)
        attn = attn.reshape(B, n_heads, w_map, h_map)   # (B, n_heads, WMAP, HMAP)

        # We'll need to transform this to the size of the outputs
        # with a differentiable function
        attn = self.up1(attn)
        attn = self.up2(attn)
        attn = self.up3(attn)

        # Now we just need to argmax the NH dimension and that will get us the labels
        #seg_map = torch.argmax(attn, dim=1)

        return attn