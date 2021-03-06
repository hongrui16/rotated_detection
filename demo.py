'''
hongrui
'''
from re import X
import cv2
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import argparse
from utiles import box2corners
from oriented_iou_loss import cal_diou, cal_giou
import math
import torchvision.models as models
from PIL import Image
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modelM3 import ModelM3
from models.modelM5 import ModelM5
from models.modelM7 import ModelM7

BATCH_SIZE = 256
NUM_EPOCH = 30 
HEIGHT = 32
WIDITH = 32

# DATA = '/content/drive/MyDrive/oriented_bbox/'

# train_dset = dict(np.load('/content/drive/MyDrive/oriented_bbox/train.npz'))
# test_dset = dict(np.load('/content/drive/MyDrive/oriented_bbox/test.npz'))
# train:
# images (10000, 32, 32) 
# boxes (10000, 4, 2) (n, (top_left, down_left, down_right, top_right), (x, y))

# test:
# images (1000, 32, 32)
# boxes (1000, 4, 2)
# [ ]

# idx = np.random.randint(1000)
# visualize(test_dset['images'][idx], [test_dset['boxes'][idx]])

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    # print(a, b)
    # print(zip(a, b))
    for a1, b1 in zip(a, b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

def order_points_quadrangle(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left and bottom-left coordinate, use it as an
    # base vector to calculate the angles between the other two vectors

    vector_0 = np.array(bl-tl)
    vector_1 = np.array(rightMost[0]-tl)
    vector_2 = np.array(rightMost[1]-tl)

    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
    



def xywha2points(x):
    # ?????????????????????????????????+-0.5pi;?????????????????????
    cx = x[0]; cy = x[1]; w = x[2]; h = x[3]; a = x[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=-a*180, center=(cx,cy), scale=1)
    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 
    points = np.array([[float(x0),float(y0)],[float(x1),float(y1)],[float(x2),float(y2)],[float(x3),float(y3)]])
    return points


class BoxDataSet(Dataset):
    def __init__(self, split="train", dataset_dir = None, transform=None):
        super(BoxDataSet, self).__init__()
        assert split in ["train", "test"], "split must be train or test"
        self.split = split
        self.transform = transform
        dataset = np.load(os.path.join(dataset_dir, split+".npz"))
        self.data = dataset['images'] ##(n, 32, 32)
        self.label = dataset['boxes'] ###(n, (top_left, down_left, down_right, top_right), (x, y))

    def convert_corners_to_xywha(self, corners):
        # (cx, cy), (w, h), theta 
        # print('corners.shape', corners.shape, corners)
        rect = cv2.minAreaRect(corners)
        # rect = cv2.minAreaRect(corners)
        cx, cy = rect[0]
        w, h = rect[1]
        theta = rect[2]
        a = theta / 180 * math.pi
        # if a >  0.5*math.pi: a = math.pi - a
        # if a < -0.5*math.pi: a = math.pi + a
        # if a < 0: a = math.pi + a
        return np.array([cx, cy, w, h, a])
    
    def convert_cc_to_c(self, corners):
        return corners[[0,3,2,1]]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) :
        x = self.data[index, ...]
        x = Image.fromarray(x)
        x = x.resize((WIDITH, HEIGHT))
        if self.transform:
            x = self.transform(x)
        # x = transforms.ToTensor()(np.array(x)/255)

        l = self.label[index, ...].astype(np.int32)
        l = self.convert_cc_to_c(l)
        l = self.convert_corners_to_xywha(l)

        return torch.FloatTensor(np.array(x)/255), torch.FloatTensor(l)


class CornersDataSet(Dataset):
    def __init__(self, split="train", dataset_dir = None):
        super(CornersDataSet, self).__init__()
        assert split in ["train", "test"], "split must be train or test"
        self.split = split
    
        dataset = np.load(os.path.join(dataset_dir, split+".npz"))
        self.data = dataset['images'] ##(n, 32, 32)
        self.label = dataset['boxes'] ###(n, (top_left, down_left, down_right, top_right), (x, y)), shape: (n, 4, 2)

    
    
    def convert_cc_to_c(self, corners):
        return (1/32)*corners[[0,3,2,1]]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) :
        d = self.data[index, ...]
        d = cv2.resize(d, (WIDITH, HEIGHT))

        l = self.label[index, ...].astype(np.int32)
        l = self.convert_cc_to_c(l)
        return torch.FloatTensor(d), torch.FloatTensor(l)


class RobNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, dilation=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.rotate = nn.Conv2d(128, 1, kernel_size=2, stride=1)
        # self.cls_prob = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.bbox = nn.Conv2d(128, 4, kernel_size=2, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print('0', x.size())
        x = F.relu(self.conv1(x), inplace=True)
        # print('1', x.size())
        
        x = F.relu(self.conv2(x), inplace=True)
        # print('2', x.size())

        x = F.relu(self.conv3(x), inplace=True)
        # print('3', x.size())

        x = F.relu(self.conv4(x), inplace=True)
        # print('4', x.size())

        x = F.relu(self.conv5(x), inplace=True)
        # print('5', x.size())

        # cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = self.rotate(x)
        # print('rotate 0', rotate.size())

        rotate = self.sig(rotate)
        # print('rotate', rotate.size())

        bbox = self.bbox(x)
        bbox = self.sig(bbox)
        # print('bbox', bbox.size())

        # return rotate, bbox
        out = torch.cat((bbox, rotate), dim=1)
        # print('out', out.size()) ###out torch.Size([256, 5, 1, 1])

        return out


class BobNet(nn.Module):
    def __init__(self, pretrained = True, bk_weight_filepath = None):
        super().__init__()
        self.backbone = ModelM3(pretrained = pretrained, weight_filepath = bk_weight_filepath)
        self.conv1 = nn.Conv2d(176, 128, kernel_size=3, padding = 1, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.rotate = nn.Conv2d(64, 1, kernel_size=2, stride=1)
        # self.cls_prob = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.bbox = nn.Conv2d(64, 4, kernel_size=2, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print('0', x.size())
        _, feature = self.backbone(x) ##[B, 176, 8, 8]
        ox = F.relu(self.conv1(feature), inplace=True)
        # print('1', x.size())
        
        ox = F.relu(self.conv2(ox), inplace=True)
        # print('2', x.size())

        rotate = self.rotate(ox)
        # print('rotate 0', rotate.size())

        rotate = self.sig(rotate)
        # print('rotate', rotate.size())

        bbox = self.bbox(ox)
        bbox = self.sig(bbox)
        # print('bbox', bbox.size())

        # return rotate, bbox
        out = torch.cat((bbox, rotate), dim=1)
        # print('out', out.size()) ###out torch.Size([256, 5, 1, 1])

        return out


class CornerNet(nn.Module):
    def __init__(self, pretrained = True, bk_weight_filepath = None):
        super().__init__()
        self.backbone = ModelM3(pretrained = pretrained, weight_filepath = bk_weight_filepath)
        self.conv1 = nn.Conv2d(176, 128, kernel_size=3, padding = 1, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        # self.rotate = nn.Conv2d(64, 1, kernel_size=2, stride=1)
        # self.cls_prob = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.corner = nn.Conv2d(64, 8, kernel_size=2, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print('0', x.size())
        _, feature = self.backbone(x) ##[B, 176, 8, 8]
        ox = F.relu(self.conv1(feature), inplace=True)
        # print('1', x.size())
        
        ox = F.relu(self.conv2(ox), inplace=True)
        # print('2', x.size())

        corners = self.corner(ox)
        # print('rotate 0', rotate.size())

        corners = self.sig(corners)
        # print('rotate', rotate.size())

        return corners

def parse_out(pred:torch.Tensor):
    p0 = (pred[..., 0] * 0.5) * WIDITH
    p1 = (pred[..., 1] * 0.5) * HEIGHT
    # p2 = (pred[..., 2] - 0.5) * WIDITH
    # p3 = (pred[..., 3] - 0.5) * HEIGHT
    p2 = (pred[..., 2] * 0.5) * WIDITH
    p3 = (pred[..., 3] * 0.5) * HEIGHT
    p4 = pred[..., 4] * np.pi/2
    return torch.stack([p0,p1,p2,p3,p4], dim=-1)


def parse_corners(pred:torch.Tensor):
    pred = pred[..., 0]  * WIDITH
    pred = pred[..., 1]  * HEIGHT

    return pred

def main(loss_type:str="giou", enclosing_type:str="aligned", dataset_dir:str=None, batchsize:int = 128, args = None):
    n_epoch = args.n_epoch
        
    transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        # transforms.RandomPosterize(bits=2),
        # transforms.RandomSolarize(threshold=100),
        # transforms.RandomEqualize()
    ])

    ds_train = BoxDataSet("train", dataset_dir, transform)
    ds_test = BoxDataSet("test", dataset_dir)
    ld_train = DataLoader(ds_train, batchsize, drop_last=False, shuffle=True, num_workers=4)
    ld_test = DataLoader(ds_test, batchsize, shuffle=False, num_workers=4)



    SM = nn.SmoothL1Loss(reduction='mean')
    # net = RobNet()
    net = BobNet(pretrained = True, bk_weight_filepath = 'weights/modelM3.pth')
    net.to("cuda:0")
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_batch = len(ds_train)//(batchsize)
    
    for epoch in range(1, n_epoch+1):
        # train
        net.train()
        for i, data in enumerate(ld_train, 1):
            image, label = data
            image = image.cuda()                            # (B, 32, 32)

            image = image.view([image.size()[0], -1, HEIGHT, WIDITH])       # (B, 1, 32, 32)
            # box = box.transpose(1, 2)               
            # print('image', image.size())    ##[256, 1, 32, 32])
            label = label.cuda()                        # (B, 5)    
            label = label.view([image.size()[0], -1, 5])     
            # print('label', label.size())    ##([256, 1, 5])

            optimizer.zero_grad()
            pred = net(image)                              
            pred = torch.squeeze(pred)
            pred = pred.view([image.size()[0], -1, 5])     
            # print('pred 0', pred.size())    ##([256, 1, 5])
            pred = parse_out(pred)
            # print('pred', pred.size())    ##([256, 1, 5])
            print('train, pre, gt', pred[0].detach().to('cpu').numpy(), label[0].detach().to('cpu').numpy())
            iou_loss, iou = None, None
            angle_loss = None
            if loss_type == "giou":
                iou_loss, iou = cal_giou(pred, label, enclosing_type)
            elif loss_type == "diou":
                iou_loss, iou = cal_diou(pred, label, enclosing_type)
            else:
                ValueError("unknown loss type")
            angle_loss = SM(pred[:,:,4],label[:,:,4])
            iou_loss = torch.mean(iou_loss)
            loss = iou_loss + 0.33*angle_loss
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                iou_mask = (iou > 0).float()
                mean_iou = torch.sum(iou) / (torch.sum(iou_mask) + 1e-8)
                print("[Epoch %d: %d/%d] train loss: %.4f  mean_iou: %.4f"
                    %(epoch, i, num_batch, loss.detach().cpu().item(), mean_iou.detach().cpu().item()))
        lr_scheduler.step()

        # validate
        net.eval()
        aver_loss = 0
        aver_mean_iou = 0
        with torch.no_grad():
            for i, data in enumerate(ld_test, 1):
                image, label = data
                image = image.cuda()                           
                image = image.view([image.size()[0], -1, HEIGHT, WIDITH])  
                # box = box.view([BATCH_SIZE, -1, 4*2])        
                # box = box.transpose(1, 2)                    
                label = label.cuda()                         
                label = label.view([image.size()[0], -1, 5])      
                
                pred = net(image)                              
                # pred = pred.transpose(1,2)                  
                pred = torch.squeeze(pred)
                pred = pred.view([image.size()[0], -1, 5])     
                pred = parse_out(pred)
                # print()
                print('val, pre, gt', pred[0].detach().to('cpu').numpy(), label[0].detach().to('cpu').numpy())
                iou_loss, iou = None, None
                angle_loss = None
                if loss_type == "giou":
                    iou_loss, iou = cal_giou(pred, label, enclosing_type)
                elif loss_type == "diou":
                    iou_loss, iou = cal_diou(pred, label, enclosing_type)
                else:
                    ValueError("unknown loss type")
                angle_loss = SM(pred[:,:,4],label[:,:,4])
                iou_loss = torch.mean(iou_loss)
                loss = iou_loss + 0.33*angle_loss
                aver_loss += loss.cpu().item()
                iou_mask = (iou > 0).float()
                mean_iou = torch.sum(iou) / (torch.sum(iou_mask) + 1e-8)
                aver_mean_iou += mean_iou.cpu().item()
        print("... validate epoch %d ..."%epoch)
        n_iter = len(ds_test)/batchsize
        print("val: average loss: %.4f"%(aver_loss/n_iter))
        print("val: average iou: %.4f"%(aver_mean_iou/n_iter))
        # print("..............................")
        print()


def YOLO_COR(loss_type:str="giou", enclosing_type:str="aligned", dataset_dir:str=None, batchsize:int = 128, args = None):
    n_epoch = args.n_epoch
    ds_train = CornersDataSet("train", dataset_dir)
    ds_test = CornersDataSet("test", dataset_dir)
    ld_train = DataLoader(ds_train, batchsize, drop_last=False, shuffle=True, num_workers=4)
    ld_test = DataLoader(ds_test, batchsize, shuffle=False, num_workers=4)
    
    # net = RobNet()
    net = CornerNet(pretrained = True, bk_weight_filepath = 'weights/modelM3.pth')
    net.to("cuda:0")
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_batch = len(ds_train)//(batchsize)
    SmoothL1Loss = nn.SmoothL1Loss()

    for epoch in range(1, n_epoch+1):
        # train
        net.train()
        for i, data in enumerate(ld_train, 1):
            image, label = data
            image = image.cuda()                            # (B, 32, 32)

            image = image.view([image.size()[0], -1, HEIGHT, WIDITH])       # (B, 1, 32, 32)
            # box = box.transpose(1, 2)               
            # print('image', image.size())    ##[256, 1, 32, 32])
            label = label.cuda()                        # (B, 5)    
            label = label.view([image.size()[0], -1, 8])     
            # print('label', label.size())    ##([256, 1, 5])

            optimizer.zero_grad()
            pred = net(image)                              
            pred = torch.squeeze(pred)
            pred = pred.view([image.size()[0], -1, 8])     
            # print('pred 0', pred.size())    ##([256, 1, 5])
            # pred = parse_corners(pred)
            # print('pred', pred.size())    ##([256, 1, 5])
            print(np.around(pred[0].detach().to('cpu').numpy(),3))
            print(np.around(label[0].detach().to('cpu').numpy(),3))
            print()
            iou_loss, iou = None, None
            # if loss_type == "giou":
            #     iou_loss, iou = cal_giou(pred, label, enclosing_type)
            # elif loss_type == "diou":
            #     iou_loss, iou = cal_diou(pred, label, enclosing_type)
            # else:
            #     ValueError("unknown loss type")
            iou_loss = SmoothL1Loss(pred, label)
            iou_loss = torch.mean(iou_loss)
            iou_loss.backward()
            optimizer.step()

            if i%10 == 0:
                # iou_mask = (iou > 0).float()
                # mean_iou = torch.sum(iou) / (torch.sum(iou_mask) + 1e-8)
                # print("[Epoch %d: %d/%d] train loss: %.4f  mean_iou: %.4f"
                #     %(epoch, i, num_batch, iou_loss.detach().cpu().item(), mean_iou.detach().cpu().item()))
                print("[Epoch %d: %d/%d] train loss: %.4f  mean_iou: %.4f"
                    %(epoch, i, num_batch, iou_loss.detach().cpu().item(),  0))
        lr_scheduler.step()

        # validate
        net.eval()
        aver_loss = 0
        aver_mean_iou = 0
        with torch.no_grad():
            for i, data in enumerate(ld_test, 1):
                image, label = data
                image = image.cuda()                           
                image = image.view([image.size()[0], -1, HEIGHT, WIDITH])  
                # box = box.view([BATCH_SIZE, -1, 8])        
                # box = box.transpose(1, 2)                    
                label = label.cuda()                         
                label = label.view([image.size()[0], -1, 8])      
                
                pred = net(image)                              
                # pred = pred.transpose(1,2)                  
                pred = torch.squeeze(pred)
                pred = pred.view([image.size()[0], -1, 8])     
                # pred = parse_corners(pred)
                # print()
                iou_loss, iou = None, None
                # if loss_type == "giou":
                #     iou_loss, iou = cal_giou(pred, label, enclosing_type)
                # elif loss_type == "diou":
                #     iou_loss, iou = cal_diou(pred, label, enclosing_type)
                # else:
                #     ValueError("unknown loss type")
                iou_loss = SmoothL1Loss(pred, label)

                iou_loss = torch.mean(iou_loss)
                aver_loss += iou_loss.cpu().item()
                # iou_mask = (iou > 0).float()
                # mean_iou = torch.sum(iou) / (torch.sum(iou_mask) + 1e-8)
                # aver_mean_iou += mean_iou.cpu().item()
        print("... validate epoch %d ..."%epoch)
        n_iter = len(ds_test)/batchsize
        print("average loss: %.4f"%(aver_loss/n_iter))
        # print("average iou: %.4f"%(aver_mean_iou/n_iter))
        print("..............................")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="diou", help="type of loss function. support: diou or giou. [default: diou]")
    parser.add_argument("--enclosing", type=str, default="smallest", 
        help="type of enclosing box. support: aligned (axis-aligned) or pca (rotated) or smallest (rotated). [default: smallest]")
    parser.add_argument("--dataset_dir", type=str, default=None, help="input dataset dir")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size")
    parser.add_argument("--gpu", type=str, default=None, help="gpu id")
    parser.add_argument("--n_epoch", type=int, default=100, help="num of epoch")
    parser.add_argument("--mode", type=str, default='xywha', help="train mode")
    parser.add_argument("--use_angle_loss", action='store_true', default=False, help="use angle loss")
  

    flags = parser.parse_args()
    
    if not flags.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"]=flags.gpu

    if flags.mode == 'xywha':
        main(flags.loss, flags.enclosing, flags.dataset_dir, flags.batchsize, args = flags)
    elif flags.mode == 'cors':
        YOLO_COR(flags.loss, flags.enclosing, flags.dataset_dir, flags.batchsize, args = flags)
    # corners, label = create_data(200)
    # print(corners.shape)
    # print(label.shape)