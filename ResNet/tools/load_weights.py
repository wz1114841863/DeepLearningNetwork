# Document function description
#   Use ours Resnet Model load the pretrained pth which support by pytorch
import os
import torch
import torch.nn as nn

import _init_paths
from lib.models import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load pretrain weights
    model_weight_path = "../pretrained_pth/resnet34_pre.pth"
    assert os.path.exists(model_weight_path), f"file {model_weight_path} doen not exist."
    
    # option1:
    # resnet = resnet34().cuda(device=device)
    # pre_weights = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    # resnet.load_state_dict(pre_weights)
    # # change fc layer structure
    # in_channel = resnet.fc.in_features
    # resnet.fc = nn.Linear(in_channel, 5)  # replace the fc layer in the end
    
    # option2:
    resnet = resnet34(num_classes=5).cuda(device=device)
    pre_weights = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    del_key = []
    for key, _ in pre_weights.items():
        if "fc" in key:
            del_key.append(key)
    for key in del_key:
        del pre_weights[key]
    
    missing_keys, unexpected_keys = resnet.load_state_dict(pre_weights, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")  

    print(resnet)
    
if __name__ == '__main__':
    main()