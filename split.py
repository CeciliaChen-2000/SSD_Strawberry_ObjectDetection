# 数据集划分集类
import os
from sklearn.model_selection import train_test_split

image_path = os.path.join('ssd.pytorch/data/VOCdevkit/VOC2007/JPEGImages')
split_txt_path = os.path.join('ssd.pytorch/data/VOCdevkit/VOC2007/ImageSets/Main/')
image_list = os.listdir(image_path)
names = []

for i in image_list:
    names.append(i.split('.')[0])  # 获取图片名
trainval, test = train_test_split(names, test_size=0.1, shuffle=3000)  # shuffle()中是图片总数目
train, validation = train_test_split(trainval, test_size=0.1, shuffle=2700)

with open(split_txt_path+'trainval.txt', 'w') as f:
    for i in trainval:
        f.write(i + '\n')
with open(split_txt_path+'test.txt', 'w') as f:
    for i in test:
        f.write(i + '\n')
with open(split_txt_path+'val.txt', 'w') as f:
    for i in validation:
        f.write(i + '\n')
with open(split_txt_path+'train.txt', 'w') as f:
    for i in train:
        f.write(i + '\n')

print('--Success--')
