from numpy.core.fromnumeric import size
from dataLoader import DataLoader
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import default
from model import UNET
from tqdm import tqdm
from tensorflow.keras.optimizers import SGD, Adam
from dataLoader import DataLoader
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='TF2 UNET for Object Segmentation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--ih', type=int, default= 160, help='Input Image Height')
    parser.add_argument('--iw', type = int, default = 160, help = 'Input Image Width')
    parser.add_argument('--LOAD_MODEL', type= bool, default= True, help='Load model')
    parser.add_argument('--train_dir', type = str, default='carvana-image-masking-challenge/train/')
    parser.add_argument('--test_dir', type = str, default='carvana-image-masking-challenge/test/')
    parser.add_argument('--train_mask_dir', type= str, default='carvana-image-masking-challenge/train_masks/')
    return parser.parse_args()

def train():

    args = get_parser()

    dataLoader = DataLoader(args.train_dir, args.train_mask_dir)
    print('Start Loading Data Images and Masks')
    train_images, train_masks, val_images, val_masks = dataLoader.train_val_split(train_size= 0.2, valid_size=0.07)

    print('Training images','Training masks','Testing images','Testing masks',sep='-->')
    print(train_images.shape, train_masks.shape, val_images.shape, val_masks.shape, sep='-->')

    model = UNET(in_channels=3, out_channels=1)
    model.compile(optimizer=tf.keras.optimizers.SGD(),loss='categorical_crossentropy', metrics=['acc'])
    model.fit(x = train_images, y = train_masks, batch_size=args.batchsize, epochs=2, verbose=1, validation_data=(val_images,val_masks),steps_per_epoch=10)
if __name__=='__main__':
    train()




