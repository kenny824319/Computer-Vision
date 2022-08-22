from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm

from tool import load_parameters
from myModels import myResnet, myLeNet, block, ResNext50
from myDatasets import cifar10_dataset
from cfg import LeNet_cfg as cfg


# The function help you to calculate accuracy easily
# Normally you won't get annotation of test label. But for easily testing, we provide you this.
def test_result(test_loader, model, device):
    pred = []
    cnt = 0
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            pred = torch.argmax(pred, axis=1)
            cnt += (pred.eq(label.view_as(pred)).sum().item())
    
    acc = cnt / len(test_loader.dataset)
    return acc

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model type', type=str, default='ResNext50')
    parser.add_argument('--path', help='model_path', type=str, default='./Model/')
    parser.add_argument('--test_anno', help='annotaion for test image', type=str, default= './p2_data/annotations/public_test_annos.json')
    args = parser.parse_args()
    
    model_type = args.model
    path = args.path+model_type+'/best_model.pt'
    test_anno = args.test_anno
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # change your model here 

    ## TO DO ## 
    # Indicate the model you use here
    #ResNext
    if model_type == 'ResNext50':
        model = ResNext50(num_class=10, device=device)

    #myLenet
    if model_type == 'myLeNet':
        model = myLeNet(num_out=10)

    #myResnet
    if model_type == 'myResnet':
        model = myResnet(block, [5, 6, 6, 4], 10).to(device)  
    
    
    # Simply load parameters
    # print(path)
    model.load_state_dict(torch.load(path))
    model.to(device)


    with open(test_anno, 'r') as f :
        data = json.load(f)    
    
    imgs, categories = data['images'], data['categories']
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    
    test_set = cifar10_dataset(images=imgs, labels= categories, transform=test_transform, prefix = './p2_data/public_test/')
    #test_set = cifar10_dataset(images=imgs, labels= categories, transform=test_transform, prefix = './p2_data/private_test/')
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
    acc = test_result(test_loader=test_loader, model=model, device=device)
    print("accuracy : ", acc)
    
if __name__ == '__main__':
    main()