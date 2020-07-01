import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

"""Make two groups according to class labels (Malignant and Benign)"""

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  
import helpers

def get_data(path):
    img=cv2.imread(path)
    return img


# Training file directory
DATASET = os.path.join(ROOT_DIR, 'dataset')

#helpers.create_directory("{}/dataGAN/".format(DATASET))
NEW_DATASET_PATH = "{}/{}".format(DATASET, "dataGAN")
NEW_DATASET_PATH

#helpers.create_directory("{}/trainA".format(NEW_DATASET_PATH))
#helpers.create_directory("{}/trainB".format(NEW_DATASET_PATH))
#helpers.create_directory("{}/trainA_".format(NEW_DATASET_PATH))
#helpers.create_directory("{}/trainB_".format(NEW_DATASET_PATH))


# Learn mapping from normal lesions to maligant lesions
NORMAL_FOLDER = "{}/trainA/".format(NEW_DATASET_PATH)
COVID_FOLDER = "{}/trainB/".format(NEW_DATASET_PATH)
COVID_FOLDER_ = "{}/trainB_/".format(NEW_DATASET_PATH)
PNEUMONIA_FOLDER = "{}/trainA_/".format(NEW_DATASET_PATH)

COVID_FOLDER, NORMAL_FOLDER, COVID_FOLDER_, PNEUMONIA_FOLDER


# IMAGES PATH
TRAINING_IMAGES = os.path.join(DATASET, 'data', 'Training_Data_forGAN')
TRAINING_IMAGES_ = os.path.join(DATASET, 'data', 'Training_Data_forGAN_Pneumonia')

# GROUND TRUTH PATH
TRAINING_GT = os.path.join(DATASET, 'data', 'Training_GroundTruth_forGAN.csv')
TRAINING_GT_ = os.path.join(DATASET, 'data', 'Training_GroundTruth_forGAN_Pneumonia.csv')

# Read the metadata
TRAINING_META = pd.read_csv(TRAINING_GT, sep=',', names=["FILENAME", "CLASS"])
TRAINING_META_ = pd.read_csv(TRAINING_GT_, sep=',', names=["FILENAME", "CLASS"])

# filenames and gts
filenames = TRAINING_META['FILENAME'].values
gt = TRAINING_META['CLASS'].values
    
filenames_ = TRAINING_META_['FILENAME'].values
gt_ = TRAINING_META_['CLASS'].values

# convert string labels to numeric values
labels = []
for s in gt:
    if s == "NORMAL" or s == 0.0 :
        labels.append(0)
    if s == "COVID-19" or s == 1.0:
        labels.append(1)

labels_ = []
for s in gt_:
    if s == "PNEUMONIA" or s == 2.0 :
        labels_.append(2)
    if s == "COVID-19" or s == 1.0:
        labels_.append(1)

# save in folders
number = 0  
for f, l in tqdm(zip(filenames[:], labels[:])):
    f = "{}/{}.jpg".format(TRAINING_IMAGES, f)
    img = get_data(f)
    
    if l == 0.0:
            cv2.imwrite(NORMAL_FOLDER + str(number) + ".jpeg", img)
            img=None
    elif l == 1.0:
        cv2.imwrite(COVID_FOLDER + str(number) + ".jpeg", img)
        img=None
    number+=1

number_ = 0  
for f, l in tqdm(zip(filenames_[:], labels_[:])):
    f = "{}/{}.jpg".format(TRAINING_IMAGES_, f)
    img = get_data(f)
    
    if l == 2.0:
            cv2.imwrite(PNEUMONIA_FOLDER + str(number_) + ".jpeg", img)
            img=None
    elif l == 1.0:
        cv2.imwrite(COVID_FOLDER_ + str(number_) + ".jpeg", img)
        img=None
    number_+=1
    
print("Done!")
