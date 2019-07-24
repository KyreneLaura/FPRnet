import numpy as np
from PIL import Image, ImageChops
from torchvision import transforms
import torch
import sys
import torchvision.datasets as datasets
import torch.utils.data as data
import gc


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        print(" SYSUData")
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # RGB format
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        del train_color_image
        del train_thermal_image
        gc.collect()

    def __getitem__(self, index):
        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        target1 = int(target1)
        target2 = int(target2)
        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (224,224)):
        print("SYSU Test")
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        del test_image
        del i
        del img
        del pix_array
        gc.collect()

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def load_data(input_data_path ):
    print("local in train", )
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
    del data_file_list
    gc.collect()
    return file_image, file_label
