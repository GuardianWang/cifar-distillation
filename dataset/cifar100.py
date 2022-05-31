import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchtoolbox.transform import Cutout

from multiprocessing import cpu_count


def get_data(root=r".", train=True, batch_size=4, augment_train=True):
    transform = []
    if train and augment_train:
        transform.extend([
            Cutout(),
            transforms.RandomHorizontalFlip()
        ])
    transform.extend([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform = transforms.Compose(transform)

    dataset = torchvision.datasets.CIFAR100(root=root, train=train,
                                            download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True if train else False,
                                             num_workers=cpu_count())

    return dataset, dataloader


def get_classes(dataset):
    return dataset.classes


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test_get_data(root=r".", batch_size=4):
    def show_batch(loader):
        plt.close('all')
        # get some random training images
        dataiter = iter(loader)
        images, labels = dataiter.next()
        print(f"image shape: {images.shape}")
        print(f"labels shape: {labels.shape}")

        # print labels
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
        # show images
        imshow(torchvision.utils.make_grid(images))

    train_dataset, train_loader = get_data(root, train=True, batch_size=batch_size)
    test_dataset, test_loader = get_data(root, train=False, batch_size=batch_size)
    classes = get_classes(train_dataset)

    print("show train loader")
    show_batch(train_loader)
    print("show test loader")
    show_batch(test_loader)


if __name__ == "__main__":
    test_get_data(root="..")
