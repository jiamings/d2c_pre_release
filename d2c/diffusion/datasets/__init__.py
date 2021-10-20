import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from torch.utils.data import Subset, TensorDataset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_distill_dataset(args, config):
    if config.data.dataset == 'CIFAR10':
        dataset = torch.load(os.path.join(args.exp, 'datasets', 'cifar10_distill/dataset.pth'))
        x = dataset['x']
        y = dataset['y']
        dataset = TensorDataset(x, y)
    else:
        dataset = None
    return dataset


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config.data.dataset == 'CIFAR100':
        dataset = CIFAR100(
            os.path.join(args.exp, "datasets", "cifar100"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR100(
            os.path.join(args.exp, "datasets", "cifar100_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config.data.dataset == "CA64_32_moco_rot":
        train_samples = np.load("/atlas/u/a7b23/winter_2021/NVAE/feats_cls_celeba/train_feats_ae_moco_32_rot.npy")
        train_labels = np.zeros(len(train_samples))

        data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)
        train_samples = (train_samples - data_mean)/data_std
        print("train data shape are - ", train_samples.shape, train_labels.shape)
        print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
            np.min(train_samples), np.max(train_samples))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

        test_samples = np.load("/atlas/u/a7b23/winter_2021/NVAE/feats_cls_celeba/val_feats_ae_moco_32.npy")
        test_labels = np.zeros(len(test_samples))

        data_mean = np.mean(test_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(test_samples, axis=(0, 2, 3), keepdims=True)
        test_samples = (test_samples - data_mean)/data_std
        print("test data shape are - ", test_samples.shape, test_labels.shape)
        
        print("test data stats are - ", np.mean(test_samples), np.std(test_samples), 
            np.min(test_samples), np.max(test_samples))


        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_labels).float())

    elif config.data.dataset == "CAHQ_moco_rot_8_1":

        train_samples = np.load("/atlas/u/a7b23/winter_2021/NVAE/out_all/celeba_256_latest_feats_train.npy")
        # train_samples = np.load("../NVAE/out_all/train_feats_celeba_256.npy")
        # train_samples = np.load("../NVAE/feats_cls_celeba/train_feats_CAHQ_rot_8_1.npy")
        # train_labels = np.load("../NVAE/feats/train_labels.npy")
        train_labels = np.zeros(len(train_samples))

        data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)
        # data_mean = np.mean(train_samples, axis=0, keepdims=True)
        # data_std = np.std(train_samples, axis=0, keepdims=True)
        train_samples = (train_samples - data_mean)/data_std
        print("train data shape are - ", train_samples.shape, train_labels.shape)
        print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
            np.min(train_samples), np.max(train_samples))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

        test_samples = np.load("/atlas/u/a7b23/winter_2021/NVAE/out_all/celeba_256_latest_feats_val.npy")
        # test_samples = np.load("../NVAE/out_all/val_feats_celeba_256.npy")
        # test_samples = np.load("../NVAE/feats_cls_celeba/val_feats_CAHQ_rot_8_1.npy")
        # test_labels = np.load("../NVAE/feats/val_labels.npy")
        test_labels = np.zeros(len(test_samples))
        # test_samples = np.reshape(test_samples, [-1, 16, 8, 8])

        data_mean = np.mean(test_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(test_samples, axis=(0, 2, 3), keepdims=True)
        # data_mean = np.mean(test_samples, axis=0, keepdims=True)
        # data_std = np.std(test_samples, axis=0, keepdims=True)
        test_samples = (test_samples - data_mean)/data_std
        print("test data shape are - ", test_samples.shape, test_labels.shape)
        
        print("test data stats are - ", np.mean(test_samples), np.std(test_samples), 
            np.min(test_samples), np.max(test_samples))


        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_labels).float())

    elif config.data.dataset == "satellite":
        
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

        fname = "/atlas/u/a7b23/data/satellite/train_30classes.csv"
        test_fname = "/atlas/u/a7b23/data/satellite/test_30_64classes.csv"


        dataset = SatDataset(fname, transform = tran_transform)
        test_dataset = SatDataset(test_fname, transform = test_transform)

    elif config.data.dataset == "cifar100_moco":
        train_samples = np.load("../out_all/train_feats_cifar100.npy")
        train_labels = np.zeros(len(train_samples))
        

        data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)
        # data_mean = np.mean(train_samples, axis=0, keepdims=True)
        # data_std = np.std(train_samples, axis=0, keepdims=True)
        train_samples = (train_samples - data_mean)/data_std
        print("train data shape are - ", train_samples.shape, train_labels.shape)
        print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
            np.min(train_samples), np.max(train_samples))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

        test_samples = np.load("../out_all/val_feats_cifar100.npy")
        test_labels = np.zeros(len(test_samples))
        # test_samples = np.reshape(test_samples, [-1, 16, 8, 8])

        data_mean = np.mean(test_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(test_samples, axis=(0, 2, 3), keepdims=True)
        # data_mean = np.mean(test_samples, axis=0, keepdims=True)
        # data_std = np.std(test_samples, axis=0, keepdims=True)
        test_samples = (test_samples - data_mean)/data_std
        print("test data shape are - ", test_samples.shape, test_labels.shape)
        
        print("test data stats are - ", np.mean(test_samples), np.std(test_samples), 
            np.min(test_samples), np.max(test_samples))


        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_labels).float())

    elif config.data.dataset == "cifar10_moco":
        train_samples = np.load("../out_all/train_feats_cifar10.npy")
        train_labels = np.zeros(len(train_samples))
        

        data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)
        # data_mean = np.mean(train_samples, axis=0, keepdims=True)
        # data_std = np.std(train_samples, axis=0, keepdims=True)
        train_samples = (train_samples - data_mean)/data_std
        print("train data shape are - ", train_samples.shape, train_labels.shape)
        print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
            np.min(train_samples), np.max(train_samples))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

        test_samples = np.load("../out_all/val_feats_cifar10.npy")
        test_labels = np.zeros(len(test_samples))
        # test_samples = np.reshape(test_samples, [-1, 16, 8, 8])

        data_mean = np.mean(test_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(test_samples, axis=(0, 2, 3), keepdims=True)
        # data_mean = np.mean(test_samples, axis=0, keepdims=True)
        # data_std = np.std(test_samples, axis=0, keepdims=True)
        test_samples = (test_samples - data_mean)/data_std
        print("test data shape are - ", test_samples.shape, test_labels.shape)
        
        print("test data stats are - ", np.mean(test_samples), np.std(test_samples), 
            np.min(test_samples), np.max(test_samples))


        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_labels).float())

    elif config.data.dataset == "ffhq_moco":
        train_samples = np.load("../out_all/train_feats_ffhq_256.npy")
        train_labels = np.zeros(len(train_samples))
        

        data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)
        # data_mean = np.mean(train_samples, axis=0, keepdims=True)
        # data_std = np.std(train_samples, axis=0, keepdims=True)
        train_samples = (train_samples - data_mean)/data_std
        print("train data shape are - ", train_samples.shape, train_labels.shape)
        print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
            np.min(train_samples), np.max(train_samples))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

        test_samples = train_samples[:500]
        test_labels = np.zeros(len(test_samples))
        # test_samples = np.reshape(test_samples, [-1, 16, 8, 8])

        data_mean = np.mean(test_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(test_samples, axis=(0, 2, 3), keepdims=True)
        # data_mean = np.mean(test_samples, axis=0, keepdims=True)
        # data_std = np.std(test_samples, axis=0, keepdims=True)
        test_samples = (test_samples - data_mean)/data_std
        print("test data shape are - ", test_samples.shape, test_labels.shape)
        
        print("test data stats are - ", np.mean(test_samples), np.std(test_samples), 
            np.min(test_samples), np.max(test_samples))


        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_labels).float())

    elif config.data.dataset == "ffhq_moco_64":
        train_samples = np.load("../out_all/train_feats_ffhq.npy")
        train_labels = np.zeros(len(train_samples))
        
        data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)
        train_samples = (train_samples - data_mean)/data_std
        print("train data shape are - ", train_samples.shape, train_labels.shape)
        print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
            np.min(train_samples), np.max(train_samples))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

        test_samples = train_samples[:500]
        test_labels = np.zeros(len(test_samples))

        data_mean = np.mean(test_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(test_samples, axis=(0, 2, 3), keepdims=True)
        test_samples = (test_samples - data_mean)/data_std
        print("test data shape are - ", test_samples.shape, test_labels.shape)
        
        print("test data stats are - ", np.mean(test_samples), np.std(test_samples), 
            np.min(test_samples), np.max(test_samples))


        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_labels).float())

    elif config.data.dataset == "ffhq_moco_32":
        train_samples = np.load("../out_all/train_feats_ffhq_32.npy")
        train_labels = np.zeros(len(train_samples))
        
        data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)
        train_samples = (train_samples - data_mean)/data_std
        print("train data shape are - ", train_samples.shape, train_labels.shape)
        print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
            np.min(train_samples), np.max(train_samples))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

        test_samples = train_samples[:500]
        test_labels = np.zeros(len(test_samples))

        data_mean = np.mean(test_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(test_samples, axis=(0, 2, 3), keepdims=True)
        test_samples = (test_samples - data_mean)/data_std
        print("test data shape are - ", test_samples.shape, test_labels.shape)
        
        print("test data stats are - ", np.mean(test_samples), np.std(test_samples), 
            np.min(test_samples), np.max(test_samples))


        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_labels).float())
    
    elif config.data.dataset == "CAHQ_moco":
        train_samples = np.load("../out_all_CAHQ/train_feats_celeba_256.npy")
        train_labels = np.zeros(len(train_samples))
        
        data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)
        train_samples = (train_samples - data_mean)/data_std
        print("train data shape are - ", train_samples.shape, train_labels.shape)
        print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
            np.min(train_samples), np.max(train_samples))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

        test_samples = train_samples[:500]
        test_labels = np.zeros(len(test_samples))

        data_mean = np.mean(test_samples, axis=(0, 2, 3), keepdims=True)
        data_std = np.std(test_samples, axis=(0, 2, 3), keepdims=True)
        test_samples = (test_samples - data_mean)/data_std
        print("test data shape are - ", test_samples.shape, test_labels.shape)
        
        print("test data stats are - ", np.mean(test_samples), np.std(test_samples), 
            np.min(test_samples), np.max(test_samples))


        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_labels).float())

    elif config.data.dataset == "ring":
        ring_dataset = Ring()
        train_samples = ring_dataset.sample(5000)
        data_mean = np.mean(train_samples, axis=0)
        data_std = np.std(train_samples, axis=0)
        train_samples = (train_samples - data_mean)/data_std
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_samples).float())

        test_samples = ring_dataset.sample(1000)
        test_samples = (test_samples - data_mean)/data_std
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_samples).float())

    elif config.data.dataset == "funnel":
        ring_dataset = Funnel()
        train_samples = ring_dataset.sample(5000)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_samples).float())

        test_samples = ring_dataset.sample(1000)
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_samples).float())

    elif config.data.dataset == "banana":
        ring_dataset = Banana()
        train_samples = ring_dataset.sample(5000)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_samples).float(), torch.from_numpy(train_samples).float())

        test_samples = ring_dataset.sample(1000)
        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_samples).float(), torch.from_numpy(test_samples).float())

    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join("/atlas/u/a7b23/data/"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=False,
            )
        else:
            dataset = CelebA(
                root=os.path.join("/atlas/u/a7b23/data/"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=False,
            )

        test_dataset = CelebA(
            root=os.path.join("/atlas/u/a7b23/data/"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=False,
        )

    elif config.data.dataset == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(args.exp, "datasets", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
        )

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
