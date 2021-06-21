import numpy as np
import cv2
import os
import glob
import scipy.io as io
import math

# means, stdevs = [], []
# d_list = []

root = '/data2/cjq/ShanghaiTechRGBD/'
depth_train_list = sorted(glob.glob(os.path.join(root, 'train_data', 'train_depth', '*'+'.mat')))
depth_test_list = sorted(glob.glob(os.path.join(root, 'test_data', 'test_depth', '*'+'.mat')))
os.makedirs(os.path.join(root, 'train_data', 'train_density'))
os.makedirs(os.path.join(root, 'test_data', 'test_density'))


def process_depth(item):
    depth = io.loadmat(item)['depth']
    index1 = np.where(depth == -999)
    index2 = np.where(depth > 20000)
    depth[index1] = 30000
    depth[index2] = 30000
    depth = depth.astype(np.float32) / 20000.
    save_dir = item.replace('mat', 'npy')
    np.save(save_dir, depth)
    return depth


def train_generate():
    for depth_path in depth_train_list:
        print(depth_path)
        depth = process_depth(depth_path)

        img_path = depth_path.replace('train_depth', 'train_img').replace('DEPTH', 'IMG').replace('mat', 'png')
        img = cv2.imread(img_path)

        gt_path = depth_path.replace('train_depth', 'train_gt').replace('DEPTH', 'GT')
        gt = io.loadmat(gt_path)['point']

        density = create_dmap(img, gt, depth)
        save_dir = depth_path.replace('train_depth', 'train_density').replace('DEPTH', 'DENSITY').replace('mat', 'npy')
        np.save(save_dir, density)


def test_generate():
    for depth_path in depth_test_list:
        print(depth_path)
        depth = process_depth(depth_path)

        img_path = depth_path.replace('test_depth', 'test_img').replace('DEPTH', 'IMG').replace('mat', 'png')
        img = cv2.imread(img_path)

        bbox_path = depth_path.replace('test_depth', 'test_bbox_anno').replace('DEPTH', 'BBOX')
        bbox = io.loadmat(bbox_path)['bbox']
        # [x1, y1, x2, y2] to [x, y]
        gt = np.zeros((bbox.shape[0], 2))
        for i in range(bbox.shape[0]):
            gt[i][0] = int((bbox[i][0]+bbox[i][2])/2)
            gt[i][1] = int((bbox[i][1]+bbox[i][3])/2)

        density = create_dmap(img, gt, depth)
        save_dir = depth_path.replace('test_depth', 'test_density').replace('DEPTH', 'DENSITY').replace('mat', 'npy')
        np.save(save_dir, density)


def create_dmap(img, gtLocation, depth, beta=0.25, downscale=8.0):
    height, width, _ = img.shape
    raw_width, raw_height = width, height
    width = math.floor(width / downscale)
    height = math.floor(height / downscale)
    raw_loc = gtLocation
    gtLocation = gtLocation / downscale
    gaussRange = 25
    # kernel = GaussianKernel(shape=(25, 25), sigma=3)
    pad = int((gaussRange - 1) / 2)
    densityMap = np.zeros((int(height + gaussRange - 1), int(width + gaussRange - 1)))
    for gtidx in range(gtLocation.shape[0]):
        if 0 <= gtLocation[gtidx, 0] < width and 0 <= gtLocation[gtidx, 1] < height:
            xloc = int(math.floor(gtLocation[gtidx, 0]) + pad)
            yloc = int(math.floor(gtLocation[gtidx, 1]) + pad)
            x_down = max(int(raw_loc[gtidx, 0] - 4), 0)
            x_up = min(int(raw_loc[gtidx, 0] + 5), raw_width)
            y_down = max(int(raw_loc[gtidx, 1]) - 4, 0)
            y_up = min(int(raw_loc[gtidx, 1] + 5), raw_height)
            depth_mean = np.sum(depth[y_down:y_up, x_down:x_up]) / (x_up - x_down) / (y_up - y_down)
            kernel = GaussianKernel((25, 25), sigma=beta * 5 / depth_mean)
            densityMap[yloc - pad:yloc + pad + 1, xloc - pad:xloc + pad + 1] += kernel
    densityMap = densityMap[pad:pad + height, pad:pad + width]
    return densityMap


def GaussianKernel(shape=(3, 3), sigma=0.5):
    """
    2D gaussian kernel which is equal to MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    radius_x, radius_y = [(radius-1.)/2. for radius in shape]
    y_range, x_range = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
    h = np.exp(- (x_range*x_range + y_range*y_range) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumofh = h.sum()
    if sumofh != 0:
        h /= sumofh
    return h


train_generate()
test_generate()
