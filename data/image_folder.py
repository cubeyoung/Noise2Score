"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.mat', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF','npy',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, label, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.label = label
        self.return_paths = return_paths
        self.loader = loader
        
    def patch(self, img):
        img = self.get_patch(
                    img,
                    patch_size=128,
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
            args[0][iy:iy + ip, ix:ix + ip],
            *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
        ]

        return ret[0]    
    def __getitem__(self, index):
        path = self.imgs[index]
        img = sio.loadmat(path)
        img = img[self.label]
        img = self.patch(img)
        img = np.expand_dims(img,0)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
