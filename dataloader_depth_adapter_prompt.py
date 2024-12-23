import math
import os
import random

from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import im_size, unknown_code, fg_path, bg_path, a_path, num_valid
from utils import safe_crop


prompts_df = pd.read_csv('data/prompt.csv')
# Data augmentation and normalization for training
# Just normalization for validation
preprocess = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ]),
}

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()
with open('Combined_Dataset/Training_set/training_bg_names.txt') as f:
    bg_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_bg_names.txt') as f:
    bg_test_files = f.read().splitlines()


def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('data/mask', name)
    alpha = cv.imread(filename, 0)
    return alpha


def get_alpha_test(name):
    fg_i = int(name.split("_")[0])
    name = fg_test_files[fg_i]
    filename = os.path.join('data/mask_test', name)
    alpha = cv.imread(filename, 0)
    return alpha


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg


def process(im_name, bg_name):
    im = cv.imread(fg_path + im_name)
    a = cv.imread(a_path + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
    return composite4(im, bg, a, w, h)


def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))     # (1, 5)
    iterations = np.random.randint(1, 20)   # (1, 20)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv.dilate(alpha, kernel, iterations)
    eroded = cv.erode(alpha, kernel, iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap


# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    # target = np.where(trimap == unknown_code)
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y


# 获取对应的prompt
def get_prompt(im_name):
    prompt = prompts_df.loc[prompts_df['image_name'] == im_name, 'prompt'].values
    return prompt[0] if len(prompt) > 0 else None


class DIMDataset(Dataset):
    def __init__(self, split, args):
        self.split = split

        filename = '{}_names.txt'.format(split)
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()

        self.transformer = data_transforms[split]
        self.stage = args.stage

    def __getitem__(self, i):
        name = self.names[i]
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_files[fcount]
        bg_name = bg_files[bcount]
        img, alpha, fg, bg = process(im_name, bg_name)

        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        trimap = gen_trimap(alpha)
        x, y = random_choice(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)
        fg = safe_crop(fg, x, y, crop_size)
        bg = safe_crop(bg, x, y, crop_size)

        trimap = gen_trimap(alpha)

        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            trimap = np.fliplr(trimap)
            alpha = np.fliplr(alpha)
            fg = np.fliplr(fg)
            bg = np.fliplr(bg)

        img_stage1 = Image.fromarray(img)
        img_stage1 = preprocess(img_stage1)
        prompt = get_prompt(im_name)

        # x = torch.zeros((4, im_size, im_size), dtype=torch.float)
        img_rgb = img[..., ::-1]  # RGB
        img_rgb = transforms.ToPILImage()(img_rgb)
        img_norm = self.transformer(img_rgb)

        label = torch.from_numpy((alpha / 255.).astype(np.float32)[np.newaxis, :, :])
        # trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        fg = torch.from_numpy(fg.astype(np.float32)).permute(2, 0, 1)
        bg = torch.from_numpy(bg.astype(np.float32)).permute(2, 0, 1)

        md_mask = torch.from_numpy(np.equal(trimap, 128).astype(np.float32)[np.newaxis, :, :])

        return img_norm, label, md_mask, img, fg, bg, prompt, img_stage1

    def __len__(self):
        return len(self.names)


def gen_names():
    num_fgs = 431
    num_bgs = 43100
    num_bgs_per_fg = 100

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    valid_names = random.sample(names, num_valid)
    train_names = [n for n in names if n not in valid_names]

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


# if __name__ == "__main__":
#     gen_names()
