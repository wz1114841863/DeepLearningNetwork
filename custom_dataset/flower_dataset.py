# Document function description
#   Custom Flower Classification Dataset
import torch
from torch.utils import data
from PIL import Image


class FlowerDataset(data.Dataset):
    "Flower classification dataset"
    def __init__(self, images_path, images_class, transform=None):
        super(FlowerDataset, self).__init__()
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        if img.mode != "RGB":
            raise ValueError(f"image {self.images_path[index]} is not RGB mode.")
        
        label = self.images_class[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        
        return images, labels
    
    
if __name__ == '__main__':
    pass