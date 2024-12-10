import argparse
import logging
import os

import cv2 as cv
import numpy as np
import torch
from torch import nn

from config import im_size, epsilon, epsilon_sqr, device

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from clip.clip import clip


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--end-epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.00001, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--pretrained', type=bool, default=None, help='pretrained model')
    parser.add_argument('--stage', type=int, default=1, help='training stage：0(simple loss), 1, 2, 3')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def safe_crop(mat, x, y, crop_size=(im_size, im_size)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.uint8)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.uint8)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (im_size, im_size):
        ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_NEAREST)
    return ret


"""只计算不确定区域alpha loss"""
def alpha_prediction_loss(y_pred, y_true, md_masks, img, fg, bg):
    unknown_region_size = md_masks.sum()
    # alpha loss
    alpha_loss = torch.sqrt(((y_pred - y_true) ** 2 + epsilon_sqr))
    alpha_loss = (alpha_loss * md_masks).sum() / (unknown_region_size + epsilon)

    return alpha_loss


def dis_646_alpha_prediction_loss(y_pred, y_true, img):
    num_pixels = float(np.prod(y_true.shape))
    # alpha loss
    alpha_loss = torch.sqrt(((y_pred - y_true) ** 2 + epsilon_sqr))
    alpha_loss = alpha_loss.sum() / (num_pixels + epsilon)
    return alpha_loss


# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
#
def compute_mse(pred, alpha, trimap):
    num_pixels = float((trimap == 128).sum())
    return ((pred - alpha) ** 2).sum() / num_pixels


# compute the SAD error given a prediction and a ground truth.
#
def compute_sad(pred, alpha):
    diff = np.abs(pred - alpha)
    return np.sum(diff) / 1000


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def load_depth_anything():
    depth_ckpt = 'Depth_Anything_V2/checkpoints/depth_anything_v2_vitb.pth'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs['vitb'])
    depth_anything.load_state_dict(torch.load(depth_ckpt, map_location='cpu'))
    depth_anything = depth_anything.to(device)
    return depth_anything


def load_clip_res50_weights():
    clip_weights, pre_process = clip.load('RN50', device='cpu')
    weights_state = clip_weights.visual.state_dict()
    new_weihts_state = weights_state.copy()
    for key in list(new_weihts_state.keys()):
        if 'attnpool' in key:
            del new_weihts_state[key]
    return new_weihts_state
