import math
import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device, fg_path_test, a_path_test, bg_path_test
from dataloader_depth_adapter_prompt import data_transforms, fg_test_files, bg_test_files
from utils import compute_mse, compute_sad, AverageMeter, get_logger

from torchvision.transforms import Compose
from clip.clip_resnet50 import ModifiedResNet, model_builder
# from MiDaS_master.midas.transforms import Resize, NormalizeImage, PrepareForNet
from Depth_Anything_V2.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from utils import load_depth_anything

home_dir = 'E:/pytorch-deep-image-matting-master/'
test_dir = 'E:/pytorch-deep-image-matting-master/data/dim_data/Combined_Dataset/Test_set/comp/'
input_path = test_dir + 'image/'
tri_map_path = test_dir + 'trimap/'
GT_path = test_dir + 'alpha/'
output_path = test_dir + 'test_out'


def gen_test_names():
    num_fgs = 50
    num_bgs = 1000
    num_bgs_per_fg = 20

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    return names


def process_test(im_name, bg_name):
    # print(bg_path_test + bg_name)
    im = cv.imread(fg_path_test + im_name)
    a = cv.imread(a_path_test + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path_test + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4_test(im, bg, a, w, h)


def composite4_test(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1. - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp, a


def gen_dataset(imgdir, trimapdir):
    sample_set = []
    img_ids = os.listdir(imgdir)
    img_ids.sort()
    cnt = len(img_ids)
    cur = 1
    for img_id in img_ids:
        img_name = os.path.join(imgdir, img_id)
        trimap_name = os.path.join(trimapdir, img_id)

        assert (os.path.exists(img_name))
        assert (os.path.exists(trimap_name))

        sample_set.append((img_name, trimap_name))

    return sample_set


def main():
    torch.manual_seed(7)
    np.random.seed(7)

    transformer = data_transforms['valid']

    net_w = net_h = 518
    resize_mode = "lower_bound"
    # normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    depth_transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method=resize_mode,
                image_interpolation_method=cv.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()
        ]
    )

    names = gen_test_names()
    """dataset = gen_dataset(input_path, tri_map_path)"""      # zc


    mse_losses = AverageMeter()
    sad_losses = AverageMeter()

    # logger = get_logger()
    i = 0
    """cur = 0"""     # zc
    for name in tqdm(names):
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        # print(im_name)
        bg_name = bg_test_files[bcount]
        trimap_name = im_name.split('.')[0] + '_' + str(i) + '.png'
        # print('trimap_name: ' + str(trimap_name))

        trimap = cv.imread('data/Combined_Dataset/Test_set/Adobe-licensed images/trimaps/' + trimap_name, 0)
        # print('trimap: ' + str(trimap))

        i += 1
        if i == 20:
            i = 0

        # img, alpha, fg, bg, new_trimap = process_test(im_name, bg_name, trimap)
        img, alpha = process_test(im_name, bg_name)

    # for img_path, trimap_path in dataset:
    #     img = cv.imread(img_path)
    #     trimap = cv.imread(trimap_path)[:, :, 0]
    #     assert (img.shape[:2] == trimap.shape[:2])
    #     img_info = (img_path.split("/")[-1], img.shape[0], img.shape[1])
    #     cur += 1

        h, w = img.shape[:2]
        # mytrimap = gen_trimap(alpha)
        # cv.imwrite('images/test/new_im/'+trimap_name,mytrimap)
        new_h = min(1600, h - (h % 32))
        new_w = min(1600, w - (w % 32))
        scale_img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        depth_img = cv.cvtColor(scale_img, cv.COLOR_BGR2RGB) / 255.0
        depth_image = depth_transform({"image": depth_img})["image"]
        depth_sample = torch.from_numpy(depth_image).to(device).unsqueeze(0)

        # x = torch.zeros((1, 4, h, w), dtype=torch.float)
        img = scale_img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)  # [3, 320, 320]
        img = transformer(img)  # [3, 320, 320]
        # x[0:, 0:3, :, :] = img
        # x[0:, 3, :, :] = torch.from_numpy(new_trimap.copy() / 255.)

        # Move to GPU, if available
        # x = x.type(torch.FloatTensor).to(device)  # [1, 4, 320, 320]
        img_tensor = img.type(torch.FloatTensor).to(device)
        img_tensor = img_tensor[None, :, :, :]
        """alpha_name = os.path.join(GT_path, img_info[0])
        alpha = cv.imread(alpha_name)[:, :, 0]"""
        alpha = alpha / 255.

        with torch.no_grad():
            depth_out = depth_model.forward(depth_sample)
            depth_map = torch.nn.functional.interpolate(depth_out, size=(new_h, new_w), mode='bilinear',
                                                        align_corners=True)

            inputs = torch.cat([img_tensor, depth_map], dim=1)
            pred = model(inputs)  # [1, 4, 320, 320]

        pred = pred.cpu().numpy()[0, 0, :, :]
        # pred = pred.reshape((h, w))  # [320, 320]
        pred = cv.resize(pred, (w, h), interpolation=cv.INTER_LINEAR)

        pred[trimap == 0] = 0.0
        pred[trimap == 255] = 1.0
        cv.imwrite('images/test/out/' + trimap_name, pred * 255)
        # cv.imwrite('images/test/out/' + img_info[0], pred * 255)

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        mse_loss = compute_mse(pred, alpha, trimap)
        sad_loss = compute_sad(pred, alpha)

        # Keep track of metrics
        mse_losses.update(mse_loss.item())
        sad_losses.update(sad_loss.item())
        print("sad:{} mse:{}".format(sad_loss.item(), mse_loss.item()))
        print("avg_sad:{} avg_mse:{}".format(sad_losses.avg, mse_losses.avg))
    print("fin_sad:{} fin_mse:{}".format(sad_losses.avg, mse_losses.avg))


if __name__ == '__main__':
    depth_model = load_depth_anything()
    depth_model.eval()

    checkpoint = 'saved_model/dim/train_87_clip.pth'
    checkpoint = torch.load(checkpoint)
    model = ModifiedResNet((3, 4, 6, 3), 1024, 32)

    num_channels = 4
    if num_channels > 3:
        model_sd = model.state_dict()
        conv1_weights = model_sd['conv1.weight']

        c_out, c_in, h, w = conv1_weights.size()
        conv1_mod = torch.zeros(c_out, num_channels, h, w)
        conv1_mod[:, :3, :, :] = conv1_weights
        conv1 = model.conv1
        conv1.in_channels = num_channels
        conv1.weight = torch.nn.Parameter(conv1_mod)

        model.conv1 = conv1
        model_sd['conv1.weight'] = conv1_mod
        model.load_state_dict(model_sd)
    # model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(checkpoint, strict=True)
    # model = checkpoint['model']
    model = model.to(device)
    model.eval()

    main()
