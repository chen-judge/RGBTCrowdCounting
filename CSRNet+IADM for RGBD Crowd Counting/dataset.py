from torch.utils.data import Dataset
from image import *
from torchvision import transforms


RGB_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(
                           mean=[0.407, 0.389, 0.396],
                           std=[0.241, 0.246, 0.242]),
                   ])
depth_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(
                           mean=[0.441],
                           std=[0.329]),
                   ])


class listDataset(Dataset):
    def __init__(self, root, shape=None, train=False, seen=0,
                 batch_size=1, num_workers=20):
        if train:
            root = root*4
            random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.RGB_transform = RGB_transform
        self.depth_transform = depth_transform

        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        img, target = load_data(img_path, self.train)
        img[0] = self.RGB_transform(img[0])
        img[1] = self.depth_transform(img[1])

        return img, target

