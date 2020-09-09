import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import transforms


def getData(csv_path,mode):

    img = pd.read_csv(csv_path+'/'+mode+'_img.csv')
    label = pd.read_csv(csv_path+'/'+mode+'_label.csv')
    return np.squeeze(img.values), np.squeeze(label.values)

labelMapping = {
    "Asthma":0,
    "Bronchiectasis":1,
    "Bronchiolitis":2,
    "COPD":3,
    "Healthy":4,
    "LRTI":5,
    "Pneumonia":6,
    "URTI":7

}

class RespiratoryLoader(data.Dataset):
    def __init__(self, img_root, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.img_root = img_root
        self.img_name, self.label = getData(root,mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

        self.transform2 = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        # step1. Get the image path from 'self.img_name' and load it.
        path = self.img_root + '/' + self.mode+'/' + self.img_name[index]
        img = Image.open(path).convert('RGB')

        # step2. Get the ground truth label from self.label
        label = labelMapping[self.label[index]]


        img = self.transform2(img)
        # step4.Return processed image and label
        return img, label