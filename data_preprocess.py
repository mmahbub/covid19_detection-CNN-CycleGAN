import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  
import helpers

'''Save images and labels in numpy format'''

def crop_and_resize(img, resize_dim=256):
    img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA)
    return img

def get_data(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=crop_and_resize(img)
    return img


# Raw dataset path
DATASET = os.path.join(ROOT_DIR, 'dataset')


# Training images path
TRAINING_IMAGES = os.path.join(DATASET, 'data', 'Training_Data')
# Training images path for GAN
TRAINING_IMAGES_GAN = os.path.join(DATASET, 'data', 'Training_Data_forGAN')
# Training images path for GAN
TRAINING_IMAGES_GAN_P = os.path.join(DATASET, 'data', 'Training_Data_forGAN_Pneumonia')

# Ground truth path
TRAINING_GT = os.path.join(DATASET, 'data', 'Training_GroundTruth.csv')
# Ground truth path for GAN
TRAINING_GT_GAN = os.path.join(DATASET, 'data', 'Training_GroundTruth_forGAN.csv')
# Ground truth path for GAN
TRAINING_GT_GAN_P = os.path.join(DATASET, 'data', 'Training_GroundTruth_forGAN_Pneumonia.csv')

# Read the metadata
TRAINING_META = pd.read_csv(TRAINING_GT, sep=',', names=["FILENAME", "CLASS"])
# Read the metadata for GAN
TRAINING_META_GAN = pd.read_csv(TRAINING_GT_GAN, sep=',', names=["FILENAME", "CLASS"])
# Read the metadata for GAN
TRAINING_META_GAN_P = pd.read_csv(TRAINING_GT_GAN_P, sep=',', names=["FILENAME", "CLASS"])

# Test images path
TEST_IMAGES = os.path.join(DATASET, 'data', 'Test_Data')
# Ground truth path
TEST_GT = os.path.join(DATASET, 'data', 'Test_GroundTruth.csv')
# Read the metadata
TEST_META = pd.read_csv(TEST_GT, sep=',', names=["FILENAME", "CLASS"])


def construct_numpy(images, meta, fname, lname):
    '''
    Creates a new numpy arrays.
    INPUT
        IMAGES: 
        df:
    OUTPUT
        Numpy arrays
    '''
    # filenames and gts
    filenames = meta['FILENAME'].values
    gt = meta['CLASS'].values
    
    # convert string labels to numeric values
    labels = []
    for s in gt:
        if s == "NORMAL" or s == 0.0 :
            labels.append(0)
        if s == "COVID-19" or s == 1.0:
            labels.append(1)
        if s == "PNEUMONIA" or s == 2.0:
            labels.append(2)
            
    # all training images and labels     
    inp_feat = []
    g_t = []

    for f, l in tqdm(zip(filenames[:], labels[:])):
        f = "{}/{}.jpg".format(images, f)
        img = get_data(f)
        inp_feat.append(img)
        g_t.append(l)
        img = None

    # make nummpy arrays
    inp_feat = np.array(inp_feat)
    g_t = np.array(g_t)
    
    # one hot encoded vectors
    num_classes = 3
    g_t = np_utils.to_categorical(g_t,num_classes)

    print(inp_feat.shape, g_t.shape)
    
    ## Create directory
    #helpers.create_directory("{}/dataNumpy/".format(DATASET))
    # Save
    np.save("{}/dataNumpy/{}.npy".format(DATASET, fname), inp_feat)
    np.save("{}/dataNumpy/{}.npy".format(DATASET, lname), g_t)
    
    print("Done!")
    


if __name__ == '__main__':
    
    # Make numpy arrays
    print("Training data...")
    construct_numpy(TRAINING_IMAGES, TRAINING_META, "x_train", "y_train")
    print("Training data for CycleGAN...")
    construct_numpy(TRAINING_IMAGES_GAN, TRAINING_META_GAN, "x_train_gan", "y_train_gan")
    print("Training data for CycleGAN with Pneumonia...")
    construct_numpy(TRAINING_IMAGES_GAN_P, TRAINING_META_GAN_P, "x_train_gan_p", "y_train_gan_p")

    print("Test data...")
    construct_numpy(TEST_IMAGES, TEST_META, "x_test", "y_test")
    
    
    
    
