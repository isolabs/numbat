import torch
from torchvision import transforms

class MultiCropper():
    """ 
    Objects of this class perform random multicropping during training in the DINO framework
    """
    
    def __init__(self,   global_size=224, local_size=96, 
                         n_global_crops=2, n_local_crops=8):
        """ 
        Global size (e.g. 224) is the side length of square crops from the images
        which represent the global scale. Local size (e.g. 96) is the side length
        of square crops from the images. They should have a common factor (e.g. 16)
        
        Local size should be less than global size. The number of local sizes (e.g. 8) 
        should be larger than the number of global sizes (e.g. 2)
        
        Use padding to ensure that the image size
        is always definitely the desired size (even if the image tensor is smaller than)
        """
        
        self.global_cropper = transforms.RandomCrop(global_size, pad_if_needed=True)
        self.local_cropper  = transforms.RandomCrop(local_size,  pad_if_needed=True)
        
        self.n_global_crops = n_global_crops
        self.n_local_crops  = n_local_crops
    
    def crop(self, image_tensor):
        """ 
        Take the provided floating point tensor (representing an image) and randomly cut 
        out the desired number and sized crops. Return them as two tensors (rather than
        lists - for the purpose of batching)
        """
        
        def crop_helper(cropper, n):
            crops = []
            for i in range(n):
                crops.append(cropper(image_tensor))
            # Tensorify
            return torch.cat(crops, dim=0)
        
        global_crops = crop_helper(self.global_cropper, self.n_global_crops)
        local_crops  = crop_helper(self.local_cropper,  self.n_local_crops)
        
        # Return separately because it will be useful to do in different forward passes
        return global_crops, local_crops