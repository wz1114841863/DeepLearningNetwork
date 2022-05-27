# Document function description
#   Used to retrain resnet34
import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

import _init_paths
from lib.models import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    data_transform = {
        "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),           
            ]),
    }
    
    batch_size = 16
    number_workers = 4
    data_root = "/home/jin/wz/Code/data"  # dataset root path
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=data_transform['train'])
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # # write into json file
    # with open("./class_indices.json", 'w') as json_file:
    #     json.dump(cla_dict, json_file)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=number_workers)
    
    validate_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader =  torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=number_workers)
    print(f"using {train_num} images for training, {val_num} images for validation.")
    
    # load pretrain weights
    model_weight_path = "../pretrained_pth/resnet34_pre.pth"
    assert os.path.exists(model_weight_path), f"file {model_weight_path} doen not exist."
    
    # option1:
    resnet = resnet34()
    pre_weights = torch.load(model_weight_path, map_location="cpu")
    resnet.load_state_dict(pre_weights)
    # change fc layer structure
    in_channel = resnet.fc.in_features
    resnet.fc = nn.Linear(in_channel, 5)  # replace the fc layer in the end
    resnet.to(device=device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()
    
    # bulid an optimizer
    params = [p for p in resnet.parameters() if p.requires_grad]
    optimizer = optim.Adam(params=params, lr=0.0001)

    epochs = 10
    best_acc = 0.0
    save_path = "../result/result.pth"
    train_steps = len(train_dataloader)
    for epoch in range(epochs):
        resnet.train()
        # train
        running_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = resnet(imgs)
            loss = loss_function(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            # print statistics
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        
        # valid     
        resnet.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_imgs, val_labels = val_data
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)
                
                outputs = resnet(val_imgs)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
                val_bar.desc = f"valid epoch[{epoch + 1}/{epochs}]"
                
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))   

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(resnet.state_dict(), save_path)
    
    print("Finished Training.")
    
if __name__ == '__main__':
    main()