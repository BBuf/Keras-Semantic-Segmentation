import os
import shutil
from os.path import join

helen_root = "./data/helen_small"

train_file = join(helen_root, "train.txt")
test_file = join(helen_root, "val.txt")

image_src_dir = join(helen_root, "images")
label_src_dir = join(helen_root, "SegClassLabel")

train_jpg_dst = "data/helen_small/train_image"
train_label_dst = "data/helen_small/train_label"

test_jpg_dst = "data/helen_small/test_image"
test_label_dst = "data/helen_small/test_label"


def mk(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


mk(train_jpg_dst)
mk(train_label_dst)
mk(test_jpg_dst)
mk(test_label_dst)

f_train = open(train_file, "r")
f_test = open(test_file, "r")

f_train_content = f_train.readlines()
f_test_content = f_test.readlines()

for line in f_train_content:
    line = line[:-1]
    print(line)
    # copy image
    shutil.copy(join(image_src_dir, line + ".jpg"),
                join(train_jpg_dst, line + ".jpg"))
    # copy label
    shutil.copy(join(label_src_dir, line + ".png"),
                join(train_label_dst, line + ".png"))

for line in f_test_content:
    line = line[:-1]
    # copy image
    shutil.copy(join(image_src_dir, line + ".jpg"),
                join(test_jpg_dst, line + ".jpg"))
    # copy label
    shutil.copy(join(label_src_dir, line + ".png"),
                join(test_label_dst, line + ".png"))

f_train.close()
f_test.close()