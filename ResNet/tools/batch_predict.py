# Document function description
#   use result.pth file to predict a batch of imgs
import os
import os
import json

import torch
from PIL import Image
from torchvision import transforms

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
    imgs_root = "./pred_imgs"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    
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
    batch_size = 4
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                    class_indict[str(cla.numpy())],
                                                                    pro.numpy()))    
                
                
if __name__ == '__main__':
    main()