import pandas as pd
import os
from pathlib import Path
import shutil
import random
import numpy as np

# Sampling of Images from GitHub
FILE_PATH = Path('chestxray/metadata.csv').absolute()
IMAGE_PATH = Path('chestxray/images/').absolute()
df = pd.read_csv(FILE_PATH)

df.head()

TARGET_DIR = Path('./Dateset/Covid').absolute()
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print("Covid Folder Created")


cnt = 0

for (i, row) in df.iterrows():
    if 'COVID-19' in row['finding'] and row['view'] == 'PA':
        filename = row["filename"]
        image_path = IMAGE_PATH / filename
        target_copy_path = TARGET_DIR / filename
        shutil.copy2(image_path, target_copy_path)
        cnt += 1

print(cnt)

# Sampling of Images from Kaggle
KAGGLE_FILE_PATH = Path('chest_xray/train/NORMAL/').absolute()
TARGET_NORMAL_DIR = Path("Dateset/Normal/").absolute()


image_names = os.listdir(KAGGLE_FILE_PATH)
random.shuffle(image_names)

for i in range(142):
    image_name = image_names[i]
    image_path = KAGGLE_FILE_PATH / image_name
    target_path = TARGET_NORMAL_DIR / image_name
    shutil.copy2(image_path, target_path)

TRAIN_PATH = Path('CovidDataset/Train').absolute()
VAL_PATH = Path('CovidDataset/Test').absolute()
