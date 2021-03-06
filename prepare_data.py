import argparse
import os
import shutil
from glob import glob
from random import shuffle

def copy_file(fromFile, toDirectory, replace_names=None):
    fromFileName = fromFile.split("/")[-1]
    if replace_names:
        for name in replace_names:
            fromFileName = fromFileName.replace(name, "")
    toFile = toDirectory + fromFileName
    print("Copying from: %s to: %s" % (fromFile, toFile))
    shutil.copy2(
        fromFile,
        toFile
    )

def verify_dir_exists(dirPath):
    if not os.path.exists(dirPath):
        print("Making directory: %s" % dirPath)
        os.makedirs(dirPath)

def prepare(img_dir, seg_dir, img_target_dir, seg_target_dir, trainSize, testSize, replace_names=None):
    print("img_dir: ", img_dir)
    print("seg_dir: ", seg_dir)

    imgs = set(glob(img_dir + "*.png")).union(set(glob(img_dir + "*.jpg")))
    segs = set(glob(seg_dir + "*.png")).union(set(glob(seg_dir + "*.jpg")))
    
    print("Found %s imgs, %s segs" % (len(imgs), len(segs)))

    # Find pairs
    pairs = []
    for img_path in list(imgs):
        seg_path = seg_dir + (img_path.split("/")[-1].replace(replace_names[0], replace_names[1]) if replace_names else
                              img_path.split("/")[-1])
        
        if seg_path in segs:
            pairs.append((img_path, seg_path))
        else:
            print("Could not find segmentation image file %s" % seg_path)
    print("candidate pairs: %s" % len(pairs))

    if len(pairs) < trainSize + testSize:
        print("%s candidates not enough! need at least %s" % (len(pairs), trainSize + testSize))
        return

    # Directories
    train_img_target_dir = img_target_dir
    train_seg_target_dir = seg_target_dir
    test_img_target_dir = img_target_dir.replace("train", "test")
    test_seg_target_dir = seg_target_dir.replace("train", "test")

    # Create target dirs
    verify_dir_exists(train_img_target_dir)
    verify_dir_exists(train_seg_target_dir)
    verify_dir_exists(test_img_target_dir)
    verify_dir_exists(test_seg_target_dir)
    shuffle(pairs)

    # Copy in the atual images
    for i in range(trainSize):
       copy_file(
           fromFile=pairs[i][0],
           toDirectory=train_img_target_dir,
           replace_names=replace_names
       )
       copy_file(
           fromFile=pairs[i][1],
           toDirectory=train_seg_target_dir,
           replace_names=replace_names
       )

    for i in range(trainSize, trainSize + testSize):
        copy_file(
            fromFile=pairs[i][0],
            toDirectory=test_img_target_dir,
            replace_names=replace_names
        )
        copy_file(
            fromFile=pairs[i][1],
            toDirectory=test_seg_target_dir,
            replace_names=replace_names
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--A_imagepath", "-Ai", type=str, default="/home/alon/from_games/images/",
                        help="dataset A's image path")
    parser.add_argument("--A_segpath", "-As", type=str, default="/home/alon/from_games/labels/",
                        help="dataset A's segmentation path")
    # cp `find train/ -name "*.png"` all_train/
    parser.add_argument("--B_imagepath", "-Bi", type=str, default="/home/alon/bdd100k/bdd100k/seg/images/train/",
                        help="dataset B's image path")
    parser.add_argument("--B_segpath", "-Bs", type=str,   default="/home/alon/bdd100k/bdd100k/seg/color_labels/train/",
                        help="dataset B's segmentation path")
    parser.add_argument("--train_size", "-tr", type=int, default=5000,
                        help="number of training examples for each dataset")
    parser.add_argument("--test_size", "-te", type=int, default=500, help="number of test examples for each dataset")
    args = vars(parser.parse_args())

    trainSize = args["train_size"]
    testSize = args["test_size"]

    print("Preparing group A")
    prepare(
        img_dir=args["A_imagepath"],
        seg_dir=args["A_segpath"],
        img_target_dir="./datasets/mlisrael/trainA/",
        seg_target_dir="./datasets/mlisrael/trainA_seg/",
        trainSize=trainSize,
        testSize=testSize,
        replace_names=("_train_color", "_train_id")
    )
    print("Preparing group B")
    prepare(
        img_dir=args["B_imagepath"],
        seg_dir=args["B_segpath"],
        img_target_dir="./datasets/mlisrael/trainB/",
        seg_target_dir="./datasets/mlisrael/trainB_seg/",
        trainSize=trainSize,
        testSize=testSize,
        replace_names=(".jpg", "_train_color.png")
    )

