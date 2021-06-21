import random
import numpy as np
import cv2


def load_data(density_path, train=True):

    if train:
        rgb_path = density_path.replace('train_density', 'train_img').\
            replace('DENSITY', 'IMG').replace('npy', 'png')
        depth_path = density_path.replace('train_density', 'train_depth').\
            replace('DENSITY', 'DEPTH')
    else:
        rgb_path = density_path.replace('test_density', 'test_img'). \
            replace('DENSITY', 'IMG').replace('npy', 'png')
        depth_path = density_path.replace('test_density', 'test_depth'). \
            replace('DENSITY', 'DEPTH')

    target = np.load(density_path)

    RGB = cv2.imread(rgb_path)[..., ::-1]
    depth = np.load(depth_path)

    if train:
        crop_ratio = random.uniform(0.5, 0.8)
        target_crop_size = (int(crop_ratio * target.shape[0]), int(crop_ratio * target.shape[1]))

        dy = int(random.random() * (target.shape[0]-target_crop_size[0]))
        dx = int(random.random() * (target.shape[1]-target_crop_size[1]))
        target = target[dy:target_crop_size[0]+dy, dx:target_crop_size[1]+dx]

        img_crop_size = (target_crop_size[0]*8, target_crop_size[1]*8)

        # CSRNet *8
        dy = dy*8
        dx = dx*8

        RGB = RGB[dy:img_crop_size[0]+dy, dx:img_crop_size[1]+dx, :]
        depth = depth[dy:img_crop_size[0]+dy, dx:img_crop_size[1]+dx]

        if random.random() > 0.8:
            RGB = np.fliplr(RGB)
            depth = np.fliplr(depth)
            target = np.fliplr(target)

    depth = np.expand_dims(depth, axis=2).copy()
    target = np.expand_dims(target, axis=0).copy()
    RGB = RGB.copy()

    data = [RGB, depth]

    return data, target


def reshape_target(target, down_sample=3):
    """ Down sample GT to 1/8

    """
    height = target.shape[0]
    width = target.shape[1]
    for i in range(down_sample):
        height = int((height+1)/2)
        width = int((width+1)/2)

    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC) * (2**(down_sample*2))

    return target
