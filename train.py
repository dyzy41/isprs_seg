from __future__ import division

import os

import cv2
import torch
import re
import time
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import tqdm
import tools.transform as tr
from tools.dataloader import IsprsSegmentation
from torchvision.utils import make_grid
import tools
from tensorboardX import SummaryWriter
from inference import slide_pred
from networks.get_model import get_net
from config import *
from tools.cal_iou import evaluate
from tools.losses import get_loss
from tools.draw_pic import draw_pic
import numpy as np
from tools.utils import label_mapping, accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_list = [i for i in range(len(gpu_id.split(',')))]


def main():
    composed_transforms_train = standard_transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.5)),
        # tr.RandomResizedCrop(img_size),
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])  # data pocessing and data augumentation
    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])  # data pocessing and data augumentation

    road_train = IsprsSegmentation(base_dir=root_data, split='train', transform=composed_transforms_train)  # get data
    trainloader = DataLoader(road_train, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)  # define traindata
    road_val = IsprsSegmentation(base_dir=root_data, split='val', transform=composed_transforms_val)  # get data
    valloader = DataLoader(road_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)  # define traindata

    if use_gpu:
        model = torch.nn.DataParallel(seg_model, device_ids=gpu_list)  # use gpu to train
        model_id = 0
        if find_new_file(model_dir) is not None:
            model.load_state_dict(torch.load(find_new_file(model_dir)))
            print('load the model %s' % find_new_file(model_dir))
            model_id = re.findall(r'\d+', find_new_file(model_dir))
            model_id = int(model_id[0])
        model.cuda()
    else:
        model = seg_model
        model_id = 0
        if find_new_file(model_dir) is not None:
            model.load_state_dict(torch.load(find_new_file(model_dir)))
            # model.load_state_dict(torch.load('./pth/best2.pth'))
            print('load the model %s' % find_new_file(model_dir))
            model_id = re.findall(r'\d+', find_new_file(model_dir))
            model_id = int(model_id[0])

    criterion = get_loss(loss_type)  # define loss
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.00001, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)  # define optimizer
    writer = SummaryWriter(os.path.join(save_dir_model, 'runs'))

    f = open(os.path.join(save_dir_model, 'log.txt'), 'w')
    for epoch in range(epoches):
        model.train()
        running_loss = 0.0
        lr = adjust_learning_rate(base_lr, optimizer, epoch, model_id, power)  # adjust learning rate
        batch_num = 0
        for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data
            images, labels = data['image'], data['gt']
            i += images.size()[0]
            labels = labels.view(images.size()[0], img_size, img_size).long()
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            if model_name != 'pspnet':
                outputs = model(images)  # get prediction
            else:
                outputs, _ = model(images)
            losses = criterion(outputs, labels)  # calculate loss
            losses.backward()  #
            optimizer.step()
            running_loss += losses
            if i % 50 == 0:
                grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('image', grid_image)
                grid_image = make_grid(
                    tools.utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()),
                    3,
                    normalize=False,
                    range=(0, 255))
                writer.add_image('Predicted label', grid_image)
                grid_image = make_grid(
                    tools.utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()),
                    3,
                    normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image)
            batch_num += images.size()[0]
            # break
        print('epoch is {}, train loss is {}'.format(epoch, running_loss.item() / batch_num))
        writer.add_scalar('learning_rate', lr, epoch)
        writer.add_scalar('train_loss', running_loss / batch_num, epoch)

        if epoch % save_iter == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, '%d.pth' % (model_id + epoch)))
            val_miou, val_acc, val_f1, val_loss = eval(valloader, model, criterion, epoch)
            val_miou_true, val_acc_true, val_f1_true = image_infer(model, epoch)
            # val_miou_true, val_acc_true, val_f1_true = 0.0, 0.0, 0.0
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('val_f1', val_f1, epoch)
            writer.add_scalar('val_miou_true', val_miou_true, epoch)
            writer.add_scalar('val_acc_true', val_acc_true, epoch)
            writer.add_scalar('val_f1_true', val_f1_true, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, val_f1:{}, val_acc:{}, val_miou:{}, \
             val_f1_true:{}, val_acc_true:{}, val_miou_true:{}\n'.format(
                str(epoch), str(lr), str(running_loss.item() / batch_num), str(val_loss), str(val_f1), str(val_acc),
                str(val_miou), str(val_f1_true), str(val_acc_true), str(val_miou_true)
            )
            # note_log += cur_log
            print(cur_log)
            # draw_pic(note_log)
            f.writelines(str(cur_log))


def eval(valloader, model, criterion, epoch):
    model.eval()

    if val_visual:
        if os.path.exists(os.path.join(save_dir, 'val_visual')) is False:
            os.mkdir(os.path.join(save_dir, 'val_visual'))
        os.mkdir(os.path.join(save_dir, 'val_visual', str(epoch)))
        os.mkdir(os.path.join(save_dir, 'val_visual', str(epoch), 'gray'))
        os.mkdir(os.path.join(save_dir, 'val_visual', str(epoch), 'color'))
    with torch.no_grad():
        batch_num = 0
        val_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(valloader), ascii=True, desc="validate step"):  # get data
            images, labels, names = data['image'], data['gt'], data['name']
            i += images.size()[0]
            labels = labels.view(images.size()[0], img_size, img_size).long()
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()

            if model_name != 'pspnet':
                outputs = model(images)  # get prediction
            else:
                outputs, _ = model(images)
            vallosses = criterion(outputs, labels)
            outputs = torch.argmax(outputs, 1)
            pred = outputs.cpu().data.numpy().astype(np.int32)
            batch_num += images.size()[0]
            val_loss += vallosses.item()
            if val_visual:
                for kk in range(len(names)):
                    pred_sub = pred[kk, :, :]
                    pred_vis = label_mapping(pred_sub)
                    cv2.imwrite(os.path.join(save_dir, 'val_visual', str(epoch), 'gray', names[kk]), pred_sub)
                    cv2.imwrite(os.path.join(save_dir, 'val_visual', str(epoch), 'color', names[kk]), pred_vis)
        # import pudb;pu.db
        val_miou, val_acc, val_f1 = evaluate(os.path.join(root_data, 'label_val'), os.path.join(save_dir, 'val_visual', str(epoch), 'gray'), num_class)
        val_loss = val_loss/batch_num
    return val_miou, val_acc, val_f1, val_loss


def image_infer(model, epoch):
    model.eval()
    imgs = os.listdir(val_path)
    if val_visual:
        os.mkdir(os.path.join(save_dir, 'val_visual', str(epoch), 'gray_big'))
        os.mkdir(os.path.join(save_dir, 'val_visual', str(epoch), 'color_big'))
    for img_path in tqdm.tqdm(imgs):
        output = slide_pred(model, os.path.join(val_path, img_path), num_class, img_size, overlap)
        pred_gray = torch.argmax(output, 1)
        pred_gray = pred_gray[0].cpu().data.numpy().astype(np.int32)
        pred_vis = label_mapping(pred_gray)
        cv2.imwrite(os.path.join(save_dir, 'val_visual', str(epoch), 'gray_big', img_path), pred_gray)
        cv2.imwrite(os.path.join(save_dir, 'val_visual', str(epoch), 'color_big', img_path), pred_vis)
    val_miou_true, val_acc_true, val_f1_true = evaluate(val_gt, os.path.join(save_dir, 'val_visual', str(epoch), 'gray_big'), num_class)
    return val_miou_true, val_acc_true, val_f1_true


def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None


def adjust_learning_rate(base_lr, optimizer, epoch, model_id, power):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = base_lr * ((1-float(epoch+model_id)/num_epochs)**power)
    lr = base_lr * (power ** ((epoch + model_id) // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    seg_model = get_net(model_name, img_size)
    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    main()
