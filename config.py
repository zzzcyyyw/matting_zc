import os.path

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 320
unknown_code = 128
epsilon = 1e-6
epsilon_sqr = epsilon ** 2

num_samples = 43100
num_train = 34480
# num_samples - num_train_samples
num_valid = 16      # 8620

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

##############################################################
# Set your paths here

saved_model_path = "saved_model/dim/"
saved_adapter_path = "saved_model/adapter/"

# path to provided foreground images
fg_path = 'data/fg/'

# path to provided alpha mattes
a_path = 'data/mask/'

# Path to background images (MSCOCO)
bg_path = 'data/bg/'

# Path to folder where you want the composited images to go
out_path = 'data/merged/'

max_size = 1600
fg_path_test = 'data/fg_test/'
a_path_test = 'data/mask_test/'
bg_path_test = 'data/bg_test/'
out_path_test = 'data/merged_test/'
##############################################################

# 646数据路径
dis646_dir = 'F:/zc_study/paper_project/Image_Matting/modnet_ensemble/data/Distinctions-646/'
img_dir = os.path.join(dis646_dir, 'Train/Comp/')
mask_dir = os.path.join(dis646_dir, 'Train/GT/')
test_img_dir = os.path.join(dis646_dir, 'Test/Comp/')
test_mask_dir = os.path.join(dis646_dir, 'Test/GT/')
