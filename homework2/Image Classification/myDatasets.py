import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 
from torchvision.transforms import transforms
from PIL import Image
import json 


def get_cifar10_train_val_set(root, ratio=0.9, cv=0):
    
    # get all the images path and the corresponding labels
    with open(root, 'r') as f:
        data = json.load(f)
    images, labels = data['images'], data['categories']

    
    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]

    # apply shuffle to generate random results 
    np.random.shuffle(info)
    x = int(N*ratio) 
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


    train_image = all_images[:x]
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]
    

    
    ## TO DO ## 
    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                ## TO DO ##
                transforms.RandomAffine(degrees=20, translate=(0, 0.2), scale=(0.9, 1.1), shear=(6, 9)),
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ToTensor(),
            ])
  
    # normally, we dont apply transform to test_set or val_set
    val_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

 
  
    ## TO DO ##
    # Complete class cifiar10_dataset
    train_set, val_set = cifar10_dataset(images=train_image, labels=train_label,transform=train_transform), \
                        cifar10_dataset(images=val_image, labels=val_label,transform=val_transform)


    return train_set, val_set



## TO DO ##
# Define your own cifar_10 dataset
class cifar10_dataset(Dataset):
    def __init__(self, images, labels=None, transform=None, prefix = './p2_data/train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        img_name = self.images[idx]
        img_name = self.prefix + '/' + img_name
        image = Image.open(img_name).convert('RGB')
        if self.transform:
          image = self.transform(image)
        return image, self.labels[idx]
        
