# import PIL
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F

from config import device, grad_clip, print_freq, saved_model_path, saved_adapter_path
from dataloader_depth_adapter_prompt import DIMDataset, data_transforms, fg_test_files, bg_test_files
from utils import parse_args, AverageMeter, clip_gradient, get_logger, \
    alpha_prediction_loss, compute_sad, compute_mse, load_depth_anything
from test_depth_adapter import gen_test_names, process_test

from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode

from net_depth_adapter_prompt import model_builder

from Depth_Anything_V2.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torch.optim.lr_scheduler import MultiStepLR
from clip import clip


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def train_net(args):
    # global scheduler
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_sad = float('inf')
    writer = SummaryWriter()

    # steps = [10, 20, 30]
    # gamma = 0.1

    # Initialize / load checkpoint
    if checkpoint is None:
        model = model_builder(weight_init, args)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
            # scheduler = MultiStepLR(optimizer, milestones=steps, gamma=gamma)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # scheduler = MultiStepLR(optimizer, milestones=steps, gamma=gamma)

    else:
        model = model_builder(weight_init)
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt, strict=True)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = DIMDataset('train', args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        depth_model.eval()
        model.train()  # train mode (dropout and batchnorm is used)

        losses = AverageMeter()

        for i, (img_norm, alpha_label, md_masks, img, fg, bg, prompts, img_stage1) in enumerate(train_loader):
            # Move to GPU, if available
            img = img.type(torch.FloatTensor).to(device)  # [N, 4, 320, 320]
            img_norm = img_norm.type(torch.FloatTensor).to(device)
            fg = fg.type(torch.FloatTensor).to(device)
            bg = bg.type(torch.FloatTensor).to(device)
            md_masks = md_masks.type(torch.FloatTensor).to(device)
            alpha_label = alpha_label.type(torch.FloatTensor).to(device)  # [N, 320, 320]
            prompts = clip.tokenize(prompts).to(device)
            img_stage1 = img_stage1.type(torch.FloatTensor).to(device)

            depth_transform = transforms.Resize((518, 518), interpolation=InterpolationMode.BICUBIC)
            depth_inputs = depth_transform(img_norm)        # img_norm
            with torch.no_grad():
                depth_out = depth_model.forward(depth_inputs)
                depth_map = torch.nn.functional.interpolate(depth_out, size=img_stage1.shape[2:], mode='bilinear',
                                                            align_corners=True)

            # Forward prop.
            # stage1：只训练深度图适配器（将深度图转换为一个文本嵌入）
            if args.stage == 1:
                text_embed, prompts_embed = model(img_stage1, depth_map, prompts)

                cosine_sim = F.cosine_similarity(text_embed.reshape(-1, 768), prompts_embed.reshape(-1, 768), dim=1)
                loss = 1 - cosine_sim.mean()

            # stage2：将stage1得到的文本嵌入和图像编码器的输出一起输入解码器
            else:
                alpha_out = model(img_norm, depth_map, prompts)       # img_norm

                loss = alpha_prediction_loss(alpha_out, alpha_label, md_masks, img, fg, bg)


            # Back prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            clip_gradient(optimizer, grad_clip)

            # Update weights
            optimizer.step()

            # Keep track of metrics
            losses.update(loss.item())

            # Print status

            if i % print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                status = 'Epoch: [{0}][{1}/{2}]\t ' \
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t ' \
                    'LR {lr:.6f}'.format(epoch, i, len(train_loader), loss=losses, lr=current_lr)
                logger.info(status)

        train_loss = losses.avg
        writer.add_scalar('Train_Loss', train_loss, epoch)

        if args.stage == 1:
            adapter_name = "train_adapter_%d.pth" % (epoch+1)
            torch.save(model.state_dict(), saved_adapter_path + '/' + adapter_name)

        else:
            checkpoint_name = "train_%d.pth" % (epoch+1)
            torch.save(model.state_dict(), saved_model_path + '/' + checkpoint_name)

            sad, mse = evaluate(model, depth_model, prompts)
            if sad < best_sad:
                best_sad = sad
                print("best_sad:{} best_mse:{}".format(best_sad, mse))
                torch.save(model.state_dict(), saved_model_path + '/' + 'ckpt_best.pth')

        # scheduler.step()


def evaluate(net, depth_model, prompts):
    net.eval()
    depth_model.eval()
    transformer = data_transforms['valid']
    net_w = net_h = 518
    resize_mode = "lower_bound"
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
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()
        ]
    )

    names = gen_test_names()

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()

    # logger = get_logger()
    i = 0
    for name in tqdm(names):
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        # print(im_name)
        bg_name = bg_test_files[bcount]
        trimap_name = im_name.split('.')[0] + '_' + str(i) + '.png'

        trimap = cv2.imread('E:/Image_Matting_Zhang/data/Combined_Dataset/Test_set/Adobe-licensed images/trimaps/' + trimap_name, 0)
        # print('trimap: ' + str(trimap))

        i += 1
        if i == 20:
            i = 0

        # img, alpha, fg, bg, new_trimap = process_test(im_name, bg_name, trimap)
        img, alpha = process_test(im_name, bg_name)
        h, w = img.shape[:2]

        new_h = min(1600, h - (h % 32))
        new_w = min(1600, w - (w % 32))

        scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        depth_img = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB) / 255.0
        depth_image = depth_transform({"image": depth_img})["image"]
        depth_sample = torch.from_numpy(depth_image).to(device).unsqueeze(0)

        img_rgb = scale_img[..., ::-1]  # RGB
        img_rgb = transforms.ToPILImage()(img_rgb)  # [3, 320, 320]
        img_rgb = transformer(img_rgb)  # [3, 320, 320]

        # Move to GPU, if available
        img_tensor = img_rgb.type(torch.FloatTensor).to(device)
        img_tensor = img_tensor[None, :, :, :]
        alpha = alpha / 255.

        with torch.no_grad():
            depth_out = depth_model.forward(depth_sample)       # depth_sample
            depth_map = torch.nn.functional.interpolate(depth_out, size=(224, 224), mode='bilinear',
                                                        align_corners=True)

            pred = net(img_tensor, depth_map, prompts)  # [1, 4, 320, 320] img_tensor

        pred = pred.cpu().numpy()[0, 0, :, :]
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        pred[trimap == 0] = 0.0
        pred[trimap == 255] = 1.0

        # Calculate loss
        mse_loss = compute_mse(pred, alpha, trimap)
        sad_loss = compute_sad(pred, alpha)

        # Keep track of metrics
        mse_losses.update(mse_loss.item())
        sad_losses.update(sad_loss.item())
    return sad_losses.avg, mse_losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    depth_model = load_depth_anything()
    main()
