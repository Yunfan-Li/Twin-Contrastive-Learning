# List of augmentations based on randaugment
import random
from PIL import Image, ImageFilter, ImageOps, ImageOps, ImageEnhance
import numpy as np
import torch
from torchvision import transforms

random_mirror = True


def ShearX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def Identity(img, v):
    return img


def TranslateX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Solarize(img, v):
    return ImageOps.solarize(img, v)


def Posterize(img, v):
    v = int(v)
    return ImageOps.posterize(img, v)


def Contrast(img, v):
    return ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):
    return ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):
    return ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):
    return ImageEnhance.Sharpness(img).enhance(v)


def augment_list():
    l = [
        (Identity, 0, 1),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Rotate, -30, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.1, 0.1),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
        (Posterize, 4, 8),
        (ShearY, -0.1, 0.1),
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


class AutoAugment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (random.random()) * float(maxval - minval) + minval
            img = op(img, val)

        return img


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Augmentation:
    def __init__(
        self,
        img_size=224,
        val_img_size=256,
        s=1,
        num_aug=4,
        cutout_holes=1,
        cutout_size=75,
        blur=1.0,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        self.weak_aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size, interpolation=Image.BICUBIC, scale=(0.2, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.4 * s, 0.2 * s)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=blur),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.strong_aug = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                AutoAugment(n=num_aug),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                Cutout(n_holes=cutout_holes, length=cutout_size),
            ]
        )
        self.val_aug = transforms.Compose(
            [
                transforms.Resize(
                    (val_img_size, val_img_size), interpolation=Image.BICUBIC
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x):
        return self.weak_aug(x), self.strong_aug(x), self.val_aug(x)


def build_transform(is_train, args):
    if args.dataset == "CIFAR-10":
        augmentation = Augmentation(
            img_size=224,
            val_img_size=256,
            s=0.5,
            num_aug=4,
            cutout_holes=1,
            cutout_size=75,
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    elif args.dataset == "CIFAR-100":
        augmentation = Augmentation(
            img_size=224,
            val_img_size=256,
            s=0.5,
            num_aug=4,
            cutout_holes=1,
            cutout_size=75,
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
    elif args.dataset == "ImageNet-10" or args.dataset == "ImageNet":
        augmentation = Augmentation(
            img_size=224,
            val_img_size=256,
            s=0.5,
            num_aug=4,
            cutout_holes=1,
            cutout_size=75,
            blur=0.5,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    return augmentation if is_train else augmentation.val_aug
