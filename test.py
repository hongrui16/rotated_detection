import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def box2corners_th(box:torch.Tensor)-> torch.Tensor:
    """convert box coordinate to corners
    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha
    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated

object = [0,0,0,3,4,3,4,0]

coors = np.array([int(x) for x in object]).reshape(4,2).astype(np.int32)



def xywha2points(x):
    # 带旋转角度，顺时针正，+-0.5pi;返回四个点坐标
    cx = x[0]; cy = x[1]; w = x[2]; h = x[3]; a = x[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    # R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
    R[:2] = cv2.getRotationMatrix2D(angle=-a, center=(cx,cy), scale=1)
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

def rotated_coord(points,cX, cY, angle):
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    ones = np.ones(shape=(len(points),1))
    points_ones = np.concatenate((points,ones), axis=1)
    transformed_pts = M.dot(points_ones.T).T
    return transformed_pts



# print(coors)
# ((cx, cy), (w, h), theta) = cv2.minAreaRect(coors)
# ###  vis & debug  opencv 0度起点，顺时针为+
# img = np.zeros((300,300))
# print(cv2.minAreaRect(coors))
# # img = cv2.imread(os.path.join(img_path, os.path.splitext(icdar_file)[0][3:])+'.jpg')
# points = cv2.boxPoints(cv2.minAreaRect(coors)).astype(np.int32)
# img = cv2.polylines(img,[points],True,(0,0,255),2)	# 后三个参数为：是否封闭/color/thickness
# # plt.plot(corners1[(0, 1, 2, 3, 0), 0], corners1[(0, 1, 2, 3, 0), 1], c = 'blue')

# cv2.imshow('display box',img)
def draw_rects(rects):
# 在im画布上画矩形rect
    im = np.zeros([240, 320], dtype = np.uint8)
    cv2.polylines(im, rects, 1, 255)
    plt.imshow(im)
    plt.show()

def draw_rect(rect):
# 在im画布上画矩形rect
    im = np.zeros([240, 320], dtype = np.uint8)
    img = cv2.polylines(im, [rect], 1, 255)
    # plt.imshow(im)
    return img

def plot_images(images, titles):
    n = len(images)
    for i in range(n):
        plt.subplot(1, 3, i+1), plt.imshow(images[i]), plt.title(titles[i])
        if isinstance(images[i], np.ndarray) and images[i].ndim == 2:
            plt.gray()
    plt.show()

def rotate_rect(rect, angle):
# 输出rect旋转后的矩形四个点的坐标，angle为正及顺时针旋转，为负及逆时针旋转
    (x,y),(w,h),a = cv2.minAreaRect(rect)
    rect_r = ((x,y), (w,h), a+angle)
    return cv2.boxPoints(rect_r).astype(np.int32)

rects = []
rect = np.array([[-10, -10], [-10, 10], [10, 10], [10, -10]], dtype=np.int32)
cX, cY = 0, 0 
W, H = 20, 20
im0 = draw_rect(rect)
(x,y),(w,h),a = cv2.minAreaRect(rect)
x = int(x)
y = int(y)
w = int(w)
h = int(h)
print(cv2.minAreaRect(rect))
print()
# draw_rect(rect)
new_rect_n = rotate_rect(rect, 45)
print('new_rect_n', new_rect_n)
print(cv2.minAreaRect(new_rect_n))
print()

points = xywha2points([x,y,w,h,45])
points = points.astype(np.int32)
print('points', points)
print(cv2.minAreaRect(points))
print()

coors = rotated_coord(rect, cX, cY, 45)
coors = coors.astype(np.int32)
print('coors', coors)
print(cv2.minAreaRect(coors))
print()



box = np.array([cX, cY, W, H, 3*np.pi/4])
t_c = torch.from_numpy(box)
print(t_c.size())
t_c = t_c.unsqueeze(0)
print(t_c.size())
t_c = t_c.unsqueeze(0)
print(t_c.size())

torch_res = box2corners_th(t_c)
print(torch_res)
# im1 = draw_rect(new_rect_n)

# print(cv2.minAreaRect(new_rect_n))
# rects.append(new_rect_n)

# new_rect_p = rotate_rect(rect, 30)
# im2 = draw_rect(new_rect_p)

# print(cv2.minAreaRect(new_rect_p))
# rects.append(new_rect_p)

# images = [im0, im1, im2]
# titles = [90, -30, 30]
# plot_images(images, titles)