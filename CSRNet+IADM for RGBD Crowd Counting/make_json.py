import json
import os
import glob

root = '/data2/cjq/ShanghaiTechRGBD/'

path = os.path.join(root, 'train_data', 'train_density')
names = glob.glob(os.path.join(path, '*.npy'))
names = sorted(names)
with open('train.json', 'w') as f:
    json.dump(names, f)

path = os.path.join(root, 'test_data', 'test_density')
names = glob.glob(os.path.join(path, '*.npy'))
names = sorted(names)
with open('test.json', 'w') as f:
    json.dump(names, f)
