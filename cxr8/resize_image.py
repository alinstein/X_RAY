#!/usr/bin/python
from PIL import Image
import os, sys
from tqdm import tqdm

path = "./dataset/images/"
save_path = "./dataset/images_resized/"
dirs = os.listdir(path)
dirs_resized = os.listdir(save_path)

for item in tqdm(dirs):
    if os.path.isfile(path + item) :
        im = Image.open(path + item).convert("RGB")
        f, e = os.path.splitext(save_path + item)
        imResize = im.resize((256, 256), Image.ANTIALIAS)
        imResize.save(f + '.png', 'png', quality=100)
