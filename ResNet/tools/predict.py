# Document function description
#   use result.pth file to predict one img
import os
import json

import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image

import _init_paths
from lib.models import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # load image
    img_path = "./pred_imgs/rose01.jpg"
    assert os.path.exists(img_path), f"file {img_path} does not exist."
    img = Image.open(img_path)
    fig = plt.figure()
    plt.imshow(img)
    
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    
    # read class indict
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), f"file {json_path} does not exist."
    with open(json_path, 'r', encoding="UTF-8") as fp:
        class_indict = json.load(fp)
    
    # create model
    model = resnet34(num_classes=5).to(device=device)
    # load trained weights
    weights_path = "../result/result.pth"
    assert os.path.exists(weights_path), f"file {weights_path} does not exist."
    model.load_state_dict((torch.load(weights_path, map_location=lambda storage, loc: storage)))
    
    # prediction
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        output = torch.squeeze(model(img)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = f"class: {class_indict[str(predict_cla)]} prob: {predict[predict_cla].numpy():.3}"
    plt.title(print_res)
    for i in range(len(predict)):
        print(f"class: {class_indict[str(i)] :10}   prob: {predict[i].numpy():.3}")
    
    plt.show()
    
    
if __name__ == '__main__':
    main()