import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
class DataLoader():
    def __init__(self, image_dir, mask_dir, transform = None):
        super(DataLoader, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def train_val_split(self, train_size, valid_size):
        train_size = int(len(self.images)* train_size)
        valid_size = int(len(self.masks)* valid_size)

        train_images = self.images[:train_size]
        val_images = self.images[train_size:valid_size+train_size]

        train_masks = self.masks[:train_size]
        val_masks = self.masks[train_size:valid_size+train_size]

        for idx in tqdm(range(len(train_images))):
            train_dir =os.path.join(self.image_dir, self.images[idx])
            train_mask_dir = os.path.join(self.mask_dir, self.masks[idx])

            train_images[idx] = np.array(Image.open(train_dir).convert("RGB"))
            train_masks[idx] = np.array(Image.open(train_mask_dir).convert("L"), dtype= np.float32)

        for idx in tqdm(range(len(val_images))):
            val_dir =os.path.join(self.image_dir, self.images[idx])
            val_mask_dir = os.path.join(self.mask_dir, self.masks[idx])

            val_images[idx] = np.array(Image.open(val_dir).convert("RGB"))
            val_masks[idx] = np.array(Image.open(val_mask_dir).convert("L"), dtype= np.float32)

        if self.transform is not None:
            aug = ImageDataGenerator(rotation_range=0.2, zoom_range=0.2, rescale= 1/255.)
            aug.flow(train_images, y=train_masks)
        
        return np.array(train_images, dtype=np.float32), np.array(train_masks,dtype=np.float32), np.array(val_images,dtype=np.float32), np.array(val_masks,dtype=np.float32)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg','_mask.gif'))
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype= np.float32)
        mask[mask==255.0]=1.0
        return image, mask


