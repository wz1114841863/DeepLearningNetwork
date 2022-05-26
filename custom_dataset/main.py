# Document function description
#   Check the Correctness of custom flower dataset
import os
from random import shuffle
import torch
from torchvision import transforms
from flower_dataset import FlowerDataset
from utils import read_split_data, plot_data_loader_image


data_root = "/home/jin/wz/Code/data/flower_photos"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_root)
    
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_data_set = FlowerDataset(train_images_path, train_images_label, data_transform["train"])    
    
    batch_size = 8
    num_workers = 4
    print(f'Using {num_workers} dataloader workers')
    train_dataloader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True, 
                                        num_workers=num_workers, collate_fn=train_data_set.collate_fn)
    # plot_data_loader_image(train_dataloader)
    
    for step, data in enumerate(train_dataloader):
        images, labels = data
        
    
if __name__ == '__main__':
    main()