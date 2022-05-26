# Document function description
#   Some tool functions
import os
import os.path as osp
import sys
import json
import pickle
import random
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def read_split_data(root, val_rate=0.2, plot_img=False):
    random.seed(0)
    assert osp.exists(root), f"data root path {root} does not exist."
    
    # Traverse folders, a folder corresponds to a category
    flower_class = [cla for cla in os.listdir(root) if osp.isdir(osp.join(root, cla))]
    flower_class.sort()
    # generate categories and corrresponding index
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    json_path = "/home/jin/wz/Code/custom_dataset/class_indics.json"
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)
    
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    suppported = [".jpg", ".JPG", ".png", ".PNG"]
    
    for cla in flower_class:
        cla_path = osp.join(root, cla)
        # get all images path of current class 
        imgs = [osp.join(cla_path, img_name) for img_name in os.listdir(cla_path)] 
                # if img_name.split('.')[-1] in suppported]
        # get current class index
        img_class = class_indices[cla]
        # record number of images in current class
        every_class_num.append(len(imgs))
        # Randomly split data according to the val_rate
        val_path = random.sample(imgs, k=int(len(imgs) * val_rate))
        # move img
        for img_path in imgs:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(img_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(img_class)
    
    print(f"{sum(every_class_num)} valid imgs found in the dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")
    
    if plot_img:
        figure = plt.figure()
        # Draw a histogram of the number of each category
        plt.bar(flower_class, every_class_num)
        # add num label in histogram
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel("image class")
        plt.ylabel("flower class distribution.")
        plt.show()
    
    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)
    
    json_path = "/home/jin/wz/Code/custom_dataset/class_indics.json"
    assert osp.exists(json_path), f"{json_path} does not exist."
    with open(json_path, 'r', encoding="UTF-8") as fp:
        class_indices = json.load(fp)
        
    for data in data_loader:
        images, labels = data
        figure = plt.figure()
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # reverse normalize operation
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # remove x-axis scale
            plt.yticks([])  # remove y-axis scale
            plt.imshow(img.astype('uint8'))
        plt.show()  
        plt.close()          


def write_pickle(list_info, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(list_info, fp)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list            
            
            
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


if __name__ == '__main__':
    data_root = "/home/jin/wz/Code/data/flower_photos"
    read_split_data(root=data_root, plot_img=True)