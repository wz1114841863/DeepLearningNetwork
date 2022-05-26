# Document function description
#  divide the flower_photos into train and valid randomly
from genericpath import exists, isfile
import os
import shutil
import random
from tqdm import tqdm


def make_folder(file_path):
    if os.path.exists(file_path) and os.path.isdir():
        for file_name in os.listdir(file_path):
            os.remove(os.path.join(file_path, file_name))
    else:
        os.makedirs(file_path)


def main():
    # ensure the result can be resume
    random.seed(0)
    # val / all
    split_rate = 0.1
    # get source img
    source_data = "./flower_photos"
    assert os.path.exists(source_data), f"path {source_data} does not exist."
    
    flower_class = [cla for cla in os.listdir(source_data) if os.path.isdir(os.path.join(source_data, cla))]
    print(flower_class)
    
    # make dir for train / val
    train_root = "./train"
    make_folder(train_root)
    for cla in flower_class:
        make_folder(os.path.join(train_root, cla))
        
    val_root = "./val"
    for cla in flower_class:
        make_folder(os.path.join(val_root, cla))
    
    for cla in flower_class:
        cla_path = os.path.join(source_data, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # Get random index
        eval_index = random.sample(images, k=int(num * split_rate))
        for image in tqdm(images):
            image_path = os.path.join(cla_path, image)
            if image in eval_index:
                new_path = os.path.join(val_root, cla)
            else:
                new_path = os.path.join(train_root, cla)
            shutil.copy(image_path, new_path)
            
    print("finish!")
    
    
if __name__ == '__main__':
    main()