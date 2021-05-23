
import os
import glob
import math

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms

def dataset_statistics(dataset, rgb=False):
    
    # We'll create a pandas dataset and get the stats from there
    data = []
    
    labelled = "label" in dataset[0].keys()
    
    for i, inst in tqdm.tqdm(enumerate(dataset), total=(len(dataset))):
        # Get the image information
        im = inst['image'].numpy()
        dims = im.shape
        channels = dims[0]
        width    = dims[1]
        height   = dims[2]
        name = inst['filename']
        im_max = np.max(im)
        im_min = np.min(im)
        
        row = [name, channels, width, height, im_min, im_max]
        
        # Also want image mean and std
        if rgb:
            mean_r = np.mean(im[0])
            mean_g = np.mean(im[1])
            mean_b = np.mean(im[2])
            std_r = np.std(im[0])
            std_g = np.std(im[1])
            std_b = np.std(im[2])
            row.extend([mean_r, mean_g, mean_b, std_r, std_g, std_b])
        
        # And label information if it exists
        if labelled:
            label = inst['label']
            row.append(label)
            
        data.append(row)
     
    # Initialise the data frame object
    cols = ["name", "channels", "width", "height", "min value", "max value"]
    if labelled:
        cols.append("label")
    if rgb:
        cols.extend(["mean r", "mean g", "mean b", "std r", "std g", "std b"])
    df = pd.DataFrame(data, columns=cols)
    
    #print(df)
    
    # Return statistics
    print(df.describe())
    df.hist()

class TransformToFloat(object):
    """
    Need to do this before normalization, do it as a transform for
    ease
    """
    def __call__(self, image):
        return image.float()
    
class TransformNormalizeMars32k(object):
    """
    Hard code the normalization values into this transform pipeline
    """
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[152.275771, 128.002617, 90.875876], 
                                               std=[29.826359, 27.273408, 22.747010])
        
    def __call__(self, image):
        return self.normalize(image)


class DatasetMars32k(Dataset):
    def __init__(self, fp_root, transform=None):
        """ 
        Reads all the image file names in the provided dataset folder. Note that there are
        no labels here
        """
        
        self.fp_root   = fp_root
        self.transform = transform
        self.filenames = [ os.path.basename(fp) for fp in glob.glob(f"{fp_root}/*.jpg") ]
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """ 
        An instance in this dataset is just an image. This returns that image with the
        desired transforms + the filename
        """
        
        # Load image
        image = read_image(f"{self.fp_root}/{self.filenames[idx]}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return as a dict
        instance = {
            'image': image,
            'filename': self.filenames[idx],
        }
        
        # TODO HDF5 / bmp efficiency
        return instance

def get_mars32k_train_test_dataloaders(filepath, data_proportion, train_test_ratio, batch_size, perform_shuffle):
    """ 
    Get the train and test dataloaders for the mars32k dataset
    """
    
    # This is our simple preprocessing chain
    transform = transforms.Compose([
        TransformToFloat(),
        TransformNormalizeMars32k()
    ])
    # Apply these transformations and load
    dataset = DatasetMars32k(filepath, transform=transform)

    # Downsize the amount of data being used if appropriate
    data_downsize = math.floor( len(dataset) * data_proportion )
    dataset = torch.utils.data.Subset(dataset, range(data_downsize))
    
    # Make the split of data
    n_train = math.floor( len(dataset) * train_test_ratio )
    n_test  = ( len(dataset) - n_train )
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    # Convert to dataloaders
    batch_size = batch_size
    shuffle = perform_shuffle
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader
    
class DatasetMSL(Dataset):
    def __init__(self, fp_root, fn_pairs, transform=None):
        """ 
        Reads all the image files in the provided dataset folder. Also reads the label
        information
        """
        
        # Save what was provided
        self.fp_root   = fp_root
        self.transform = transform
        
        # Load the names of the classes for this dataset
        with open(f"{self.fp_root}/{fn_msl_label_names}", 'r') as f:
            self.label_name_map = [' '.join(ss.split()[1:]) for ss in f.readlines()]
        
        # Load the filename / label pairs from the provided file
        pairs = np.genfromtxt(f"{self.fp_root}/{fn_pairs}", dtype=None, encoding=None)
        self.filenames = [p[0] for p in pairs]
        self.labels    = [p[1] for p in pairs]
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """ 
        An instance in this dataset is the image, label, and associated info
        """
        
        # Load image
        image = read_image(f"{self.fp_root}/{self.filenames[idx]}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return as a dict
        instance = { 
            'image': image,
            'filename': self.filenames[idx],
            'label': self.labels[idx],
            'label_name': self.label_name_map[self.labels[idx]]
        }
        
        # TODO HDF5 / bmp efficiency
        return instance
    