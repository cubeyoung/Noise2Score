import os.path
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sio
import numpy as np

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'trainA')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'trainB')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.max = 0.435281
        self.min = -0.094310045
        self.max_min = self.max - self.min
        
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
    def patch(self, img):
        img = self.get_patch(
                    img,
                    patch_size=256,
                    scale=1,
                    multi=False,
                    input_large=False
                )
        #img = common.augment(img)
        return img
    
    def get_patch(self,*args, patch_size=96, scale=2, multi=False, input_large=False):
        ih, iw = args[0].shape[:2]

        if not input_large:
            p = scale if multi else 1
            tp = p * patch_size
            ip = tp // scale
        else:
            tp = patch_size
            ip = patch_size

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        if not input_large:
            tx, ty = scale * ix, scale * iy
        else:
            tx, ty = ix, iy

        ret = [
            args[0][iy:iy + ip, ix:ix + ip,:],
            *[a[ty:ty + tp, tx:tx + tp,:] for a in args[1:]]
        ]

        return ret[0]  
    def augment(self,*args, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        
        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            return img

        return [_augment(a) for a in args]  
    
    def np2Tensor(self,*args):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()

            return tensor

        return [_np2Tensor(a) for a in args]    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = np.expand_dims(np.load(A_path),2)        
        B_img = np.expand_dims(np.load(B_path),2)
        A_img = (A_img -self.min)/self.max_min
        B_img = (B_img - self.min)/self.max_min
        A = self.patch(A_img)       
        B = self.patch(B_img)
        A,B = self.augment(A,B)
        A = self.np2Tensor(A)[0]
        B = self.np2Tensor(B)[0]

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
