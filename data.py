import torchvision
from typing import Any, Callable, Optional
from PIL import Image
from torchvision.datasets.folder import default_loader
from transforms import build_transform
from torch.utils import data


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(CIFAR10, self).__init__(
            root, train, transform, target_transform, download
        )
        self.train = train

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target, index
        else:
            return img, target, index + 50000


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class ImageFolder(torchvision.datasets.DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, index) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


def build_dataset(type, args):
    is_train = type == "train"
    transform = build_transform(is_train, args)
    root = args.data_path

    if args.dataset == "CIFAR-10":
        dataset = data.ConcatDataset(
            [
                CIFAR10(
                    root=root + "CIFAR-10",
                    train=True,
                    download=True,
                    transform=transform,
                ),
                CIFAR10(
                    root=root + "CIFAR-10",
                    train=False,
                    download=True,
                    transform=transform,
                ),
            ]
        )
    elif args.dataset == "CIFAR-100":
        dataset = data.ConcatDataset(
            [
                CIFAR100(
                    root=root + "CIFAR-100",
                    train=True,
                    download=True,
                    transform=transform,
                ),
                CIFAR100(
                    root=root + "CIFAR-100",
                    train=False,
                    download=True,
                    transform=transform,
                ),
            ]
        )
    elif args.dataset == "ImageNet-10":
        dataset = ImageFolder(root=root + "ImageNet-10", transform=transform)
    elif args.dataset == "ImageNet":
        dataset = ImageFolder(root=root + "ImageNet/train", transform=transform)

    print(dataset)

    return dataset
