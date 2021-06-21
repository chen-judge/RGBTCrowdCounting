
# CSRNet+IADM for RGBD crowd counting

We take CSRNet as backbone to develop our framework on [ShanghaiTechRGBD](https://github.com/svip-lab/RGBD-Counting) benchmark. 

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 2.7

PyTorch: 0.4.0

## Preprocessing

Generation the ground-truth density maps for training (please edit the dataset root path in the script).
```
python RGBD_GT_generation.py
```

Make data path files and edit this file to change the path to your original datasets.
```
python make_json.py
```


## Training
Edit this file for training CSRNet-based IADM model.
```
bash train.sh
```

## Testing
Edit this file for testing models.
```
bash test.sh
```

