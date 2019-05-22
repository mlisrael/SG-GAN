import os
from collections import defaultdict
from glob import glob
from multiprocessing import Pool as ProcessPool

import numpy as np
from scipy.misc import imread, toimage


num_seg_masks = 8
# vehicles: 1
# pedestrians: 2
# cyclist: 3
# roads: 4
# buildings: 5
# sky: 6
# tree: 7
# others: 0


# https://bitbucket.org/visinf/projects-2016-playing-for-data/src/6afee1a5923f452e741c9256f5fb78f2b3882ee2/label
# /initLabels.m?at=master&fileviewer=file-view-default
"""
('0,0,0', 'unlabeled')
('0,0,0', 'ego vehicle')
('0,0,0', 'rectification border')
('0,0,0', 'out of roi')
('20,20,20', 'static')
('111,74,0', 'dynamic')
('81,0,81', 'ground')
('128,64,128', 'road')
('244,35,232', 'sidewalk')
('250,170,160', 'parking')
('230,150,140', 'rail track')
('70,70,70', 'building')
('102,102,156', 'wall')
('190,153,153', 'fence')
('180,165,180', 'guard rail')
('150,100,100', 'bridge')
('150,120,90', 'tunnel')
('153,153,153', 'pole')
('153,153,153', 'polegroup')
('250,170,30', 'traffic light')
('220,220,0', 'traffic sign')
('107,142,35', 'vegetation')
('152,251,152', 'terrain')
('70,130,180', 'sky')
('220,20,60', 'person')
('255,0,0', 'rider')
('0,0,142', 'car')
('0,0,70', 'truck')
('0,60,100', 'bus')
('0,0,90', 'caravan')
('0,0,110', 'trailer')
('0,80,100', 'train')
('0,0,230', 'motorcycle')
('119,11,32', 'bicycle')
('0,0,142', 'license plate')
"""


# https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
def cityscape():
    rgb_to_maskIndex = defaultdict(int)
    rgb_to_maskPairs = [
        ((128, 64, 128), 4), # road
        ((244, 35, 232), 4), # sidewalk
        ((250, 170, 160), 4), # parking
        ((230, 150, 140), 4), # rail track
        ((70, 70, 70), 5),    # building
        ((102, 102, 156), 5), # wall
        ((190, 153, 153), 5), # fence
        ((180, 165, 180), 5), # guard rail
        ((150, 100, 100), 5), # bridge
        ((150, 120, 90), 5), # tunnel
        ((107, 142, 35), 7), # vegetation
        ((152, 251, 152), 8), # terrain
        ((70, 130, 180), 6), # sky
        ((220, 20, 60), 2), # person
        ((255, 0, 0), 2), # rider
        ((0, 0, 142), 1), # car
        ((0, 0, 70), 1), # truck
        ((0, 60, 100), 1), # bus
        ((0, 0, 90), 1), # caravan
        ((0, 0, 110), 1), # trailer
        ((0, 80, 100), 10), # train
        ((0, 0, 230), 3), # motorcycle
        ((119, 11, 32), 3), # bicycle
        ((153, 153, 153), 9), # pole & pole group
        ((250, 170, 30), 9), # traffic light
        ((220, 220, 0), 9), # traffic sign
    ]

    for k, v in rgb_to_maskPairs:
        rgb_to_maskIndex[k] = v
    return rgb_to_maskIndex

def A_maskmap():
    return cityscape()

def B_maskmap():
    return cityscape()

# Preprocess images - creating segment maps for SG-GAN
def preprocess(image_seg_tuple):
    (index, image_seg, maskmap) = image_seg_tuple
    base_name = os.path.basename(image_seg)
    print("Processing image #%s: %s" % (index, base_name))
    img = imread(image_seg)
    M, N = img.shape[:2]
    seg_class = np.zeros((M, N)).astype(np.int)
    for x in range(M):
        for y in range(N):
            seg_class[x, y] = maskmap[tuple(img[x, y, :3])]
    toimage(seg_class, cmin=0, cmax=255).save(image_seg.replace("_seg", "_seg_class"))

# Run 8 process, preprocessing images
def preprocess_master(src, maskmap):
    dst = src.replace("_seg", "_seg_class")
    if not os.path.exists(dst):
        os.makedirs(dst)
    segs = set(glob(src + "/*.png"))
    segs = list(
        (idx, image_seg, maskmap)
        for (idx, image_seg) in enumerate(segs)
    )
    pool = ProcessPool(8)
    pool.map(preprocess, segs)

if __name__ == "__main__":
    maskmapA = A_maskmap()
    preprocess_master("datasets/mlisrael/trainA_seg", maskmapA)
    
    maskmapB = A_maskmap()
    preprocess_master("datasets/mlisrael/trainB_seg", maskmapB)

