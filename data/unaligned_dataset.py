import os.path
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random

"""
Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
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

        self.phase = opt.phase
        self.serial_batches = opt.serial_batches

        BaseDataset.__init__(self, opt)
        # create a path '/path/to/data/trainA'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        # load images from '/path/to/data/trainA'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        # get the size of dataset A
        self.A_size = len(self.A_paths)

        # only read B set when training
        if self.phase == 'train':
            # create a path '/path/to/data/trainB'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
            # load images from '/path/to/data/trainB'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
            self.B_size = len(self.B_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'

        # get the number of channels of input image
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        # get the number of channels of output image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_test = get_transform(self.opt, grayscale=(output_nc == 1))

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
        # make sure index is within then range
        A_path = self.A_paths[index % self.A_size]

        if self.phase == 'train':
            # make sure index is within then range
            if self.serial_batches:
                index_B = index % self.B_size
            else:
                # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            # apply image transformation
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)

            ret = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        elif self.phase == 'test':
            A_img = Image.open(A_path).convert('RGB')
            A = self.transform_test(A_img)
            ret = {'A': A, 'A_paths': A_path}

        return ret

    def __len__(self):
        """ the total number of images in the dataset.
            return the larger of A or B for training, size of A for testing
        """
        if self.phase == 'train':
            ret = max(self.A_size, self.B_size)
        else:
            ret = self.A_size
        return ret



