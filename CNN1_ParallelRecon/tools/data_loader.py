import os
import scipy.io as scio

from torch.utils import data
from torchvision import transforms as T

def load_data_mat(image_path):
    data_in = scio.loadmat(image_path)
    input_temp = data_in['under']
    label_temp = data_in['label']
    mask = data_in['mask']

    return input_temp, label_temp, mask

def load_data_mat_test(image_path):
    data_in = scio.loadmat(image_path)
    input_temp = data_in['under']
    mask = data_in['mask']

    return input_temp, mask

class ImageFolder(data.Dataset):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, root, config, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.root = root
        self.mode = mode
        self.image_dir = os.path.join(root, mode)

        self.image_paths = list(map(lambda x: os.path.join(self.image_dir, x), os.listdir(self.image_dir)))
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image, GT, mask = load_data_mat(image_path)

        # -----To Tensor------#
        Transform = T.ToTensor()
        image = Transform(image)
        GT = Transform(GT)
        mask = Transform(mask)

        return image, GT, mask

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

class ImageFolder_test(data.Dataset):
    """Load Variaty Chinese Fonts for Iterator. """

    def __init__(self, root, config, mode='train'):
        """Initializes image paths and preprocessing module."""
        self.config = config
        self.root = root
        self.mode = mode
        self.image_dir = os.path.join(root, mode)

        self.image_paths = list(map(lambda x: os.path.join(self.image_dir, x), os.listdir(self.image_dir)))
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image, mask = load_data_mat_test(image_path)

        # -----To Tensor------#
        Transform = T.ToTensor()
        image = Transform(image)
        mask = Transform(mask)

        return image, mask

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, config, num_workers, shuffle=True,mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, config=config, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader

def get_loader_test(image_path, config, num_workers, shuffle=True,mode='train'):
    """Builds and returns Dataloader."""

    dataset = ImageFolder_test(root=image_path, config=config, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=8,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader