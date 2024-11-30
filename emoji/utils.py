import os
import numpy as np
from PIL import Image
import torchvision
import random

def get_cls2seq_table(path='./data/EMOJI/train/'):
    cls2seq = {}
    if os.path.exists('./data/EMOJI/cls2seq.txt'):
        with open('./data/EMOJI/cls2seq.txt', 'r') as f:
            for line in f:
                cls2seq[line.split(' ')[0]] = int(line.split(' ')[1])
        return cls2seq
    else:
        i = 0
        for item in os.listdir(path):
            cls2seq[item] = i
            i += 1
        with open('./data/EMOJI/cls2seq.txt', 'w') as f:
            for item in cls2seq:
                f.writelines(item + ' ' + str(cls2seq[item]) + '\n')
    return cls2seq

def get_seq2cls_table(path='./data/EMOJI/train/'):
    seq2cls = {}
    if os.path.exists('./data/EMOJI/cls2seq.txt'):
        with open('./data/EMOJI/cls2seq.txt', 'r') as f:
            for line in f:
                seq2cls[str(int(line.split(' ')[1]))] = line.split(' ')[0]
        return seq2cls
    else:
        get_cls2seq_table()
        return get_seq2cls_table()
    return seq2cls

def mv_data(cls2seq=None, path='./data/EMOJI/train'):
    X = list()
    y = dict()
    if cls2seq is None:
        cls2seq = get_cls2seq_table()
    for cls_name in os.listdir(path):
        for img_name in os.listdir(os.path.join(path, cls_name)):
            os.system('cp ' + os.path.join(path, cls_name, img_name) + ' ' + os.path.join('./data/EMOJI/train_imgs', img_name))
            y[img_name] = cls2seq[cls_name]
    with open('./data/EMOJI/train_labels/labels.csv', 'w') as f:
        f.writelines('img_name,' + 'cls_label\n')
        for item in y:
            line = item + ',' + str(y[item]) + '\n'
            f.writelines(line)
    return X, y

def scaling_data(path='./data/EMOJI/train'):

    def apply(img, scale_num=5):
        trans_list = [torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomVerticalFlip(),
                        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]
        imgs = [random.choice(trans_list)(img) for _ in range(scale_num)]
        return imgs
    y = dict()
    cls2seq = get_cls2seq_table()
    for cls_name in os.listdir(path):
        for img_name in os.listdir(os.path.join(path, cls_name)):
            single_img = Image.open(os.path.join(path, cls_name, img_name)).convert('RGB')
            scaling_imgs = apply(single_img)            
            os.system('cp ' + os.path.join(path, cls_name, img_name) + ' ' + os.path.join('./data/EMOJI/train_scale_imgs', img_name))
            y[img_name] = cls2seq[cls_name]
            for i, scaling_img in enumerate(scaling_imgs):
                scaling_img.save(os.path.join('./data/EMOJI/train_scale_imgs', img_name.replace('.png', '') + '_' + str(i) + '.png'))
                y[img_name.replace('.png', '') + '_' + str(i) + '.png'] = cls2seq[cls_name]
    with open('./data/EMOJI/train_scale_labels/labels.csv', 'w') as f:
        f.writelines('img_name,' + 'cls_label\n')
        for item in y:
            line = item + ',' + str(y[item]) + '\n'
            f.writelines(line)

def one_hot_encode(label, num_classes=50):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


if __name__ == '__main__':
    get_cls2seq_table()
    mv_data()
    print('mv_data done')
    scaling_data()