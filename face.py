import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import copy

num_of_keypoints = 7
resize_pixel = 255

def show_key_points(img, key_points):
    key_points_x = []
    key_points_y = []
    for point in key_points:
        key_points_x.append(point[0])
        key_points_y.append(point[1])
    plt.imshow(img)
    plt.scatter(x=key_points_x, y=key_points_y, c='r', s=20)
    plt.show()

def crop_bounding_box(img, item):
    img = img[item['bounding_box'][0][1]:item['bounding_box'][1][1]+1,
                  item['bounding_box'][0][0]:item['bounding_box'][1][0]+1, :]
    for point in item['key_points']:
        point[0] = point[0] - item['bounding_box'][0][0]
        point[1] = point[1] - item['bounding_box'][0][1]
    return img, item['key_points']

def resize(img, key_points):
    img = (img * 255).astype('uint8')
    temp = Image.fromarray(img).resize((resize_pixel, resize_pixel))
    img_resized = np.asarray(temp, dtype=np.float32) / 255
    width_ratio = len(img[1]) / resize_pixel
    height_ratio = len(img[0]) / resize_pixel
    for point in key_points:
        point[0] = point[0] / width_ratio
        point[1] = point[1] / height_ratio
    return img_resized, key_points

def flipping(img, key_points):
    img = np.flip(img, 1)
    for point in key_points:
        point[0] = len(img[1]) - 1 - point[0]
    key_points_flipped = key_points.copy()
    key_points_flipped[0] = key_points[3]
    key_points_flipped[1] = key_points[2]
    key_points_flipped[2] = key_points[1]
    key_points_flipped[3] = key_points[0]
    key_points_flipped[4] = key_points[5]
    key_points_flipped[5] = key_points[4]
    return img, key_points_flipped

def change_brightness(img, offset):
    offset = float(offset / 255)
    for i in range(len(img[1])):
        for j in range(len(img[0])):
            img[i, j] = img[i, j] + offset
            if img[i, j] > 1:
                img[i, j] = 1
            elif img[i, j] < 0:
                img[i, j] = 0
    return img

#root_dir = "/home/songyih/Documents/lab2/lfw"
root_dir = "E:\lfw"
save_dir = "E:\\face_cropped"

# get file list
annotation_train_path = os.path.join(root_dir, 'LFW_annotation_train.txt')
data_list = []
with open(annotation_train_path, "r") as f:
    for line in f:
        if line == "\n":
            break
        tokens = line.split("\t")   #token[0]: file name; token[1]: bounding box; token[2]: key points
        tokens[1] = tokens[1].split()
        tokens[2] = tokens[2].split()
        # key_points: [0]canthus_rr, [1]canthus_rl, [2]canthus_lr, [3]canthus_ll,
        #             [4]mouth_corner_r, [5]mouth_corner_l, [6]nose
        data_list.append({'file_name': tokens[0],
                          'bounding_box': [[int(tokens[1][0]), int(tokens[1][1])],
                                           [int(tokens[1][2]), int(tokens[1][3])]],
                          'key_points': [[float(tokens[2][0]), float(tokens[2][1])],
                                         [float(tokens[2][2]), float(tokens[2][3])],
                                         [float(tokens[2][4]), float(tokens[2][5])],
                                         [float(tokens[2][6]), float(tokens[2][7])],
                                         [float(tokens[2][8]), float(tokens[2][9])],
                                         [float(tokens[2][10]), float(tokens[2][11])],
                                         [float(tokens[2][12]), float(tokens[2][13])]]})

idx = 9998    #select picture
item = data_list[idx]
file_path = os.path.join(root_dir, item['file_name'][:-9], item['file_name'])
img = np.asarray(Image.open(file_path), dtype=np.float32) / 255
show_key_points(img, item['key_points'])
img_in = img.copy()
img_flipped, key_points_flipped = flipping(img_in, item['key_points'])
show_key_points(img_flipped, key_points_flipped)
'''
imitate getitem
'''
'''
idx = 9998    #select picture
item = data_list[idx]
file_path = os.path.join(root_dir, item['file_name'][:-9], item['file_name'])
img = np.asarray(Image.open(file_path).convert('L'), dtype=np.float32) / 255
#show_key_points(img, item['key_points'])

# crop image by bounding box
img_in = img.copy()
item_in = copy.deepcopy(item)
img_cropped, key_points_cropped = crop_bounding_box(img_in, item_in)
show_key_points(img_cropped, key_points_cropped)

# resize to 255
img_cropped_in = img_cropped.copy()
key_points_cropped_in = copy.deepcopy(key_points_cropped)
img_resized, key_points_resized = resize(img_cropped_in, key_points_cropped_in)
#show_key_points(img_resized, key_points_resized)

#flipping
img_flipped, key_points_flipped = flipping(img_resized, key_points_resized)
#show_key_points(img_flipped, key_points_flipped)

#change brightness
img_cropped_in = img_cropped.copy()
img_bright = change_brightness(img_cropped_in, 80)
show_key_points(img_bright, key_points_cropped)
'''
'''
train_cropped_out = ""
for idx in range(10000):
    print("processing picture #" + str(idx))
    item = data_list[idx]
    file_path = os.path.join(root_dir, item['file_name'][:-9], item['file_name'])
    img = np.asarray(Image.open(file_path), dtype=np.float32) / 255
    # crop image by bounding box
    img_in = img.copy()
    item_in = copy.deepcopy(item)
    img_cropped, key_points_cropped = crop_bounding_box(img_in, item_in)
    train_cropped_out += item['file_name'] + ',' + \
                         str(round(key_points_cropped[0][0], 4)) + ',' + \
                         str(round(key_points_cropped[0][1], 4)) + ',' + \
                         str(round(key_points_cropped[1][0], 4)) + ',' + \
                         str(round(key_points_cropped[1][1], 4)) + ',' + \
                         str(round(key_points_cropped[2][0], 4)) + ',' + \
                         str(round(key_points_cropped[2][1], 4)) + ',' + \
                         str(round(key_points_cropped[3][0], 4)) + ',' + \
                         str(round(key_points_cropped[3][1], 4)) + ',' + \
                         str(round(key_points_cropped[4][0], 4)) + ',' + \
                         str(round(key_points_cropped[4][1], 4)) + ',' + \
                         str(round(key_points_cropped[5][0], 4)) + ',' + \
                         str(round(key_points_cropped[5][1], 4)) + ',' + \
                         str(round(key_points_cropped[6][0], 4)) + ',' + \
                         str(round(key_points_cropped[6][1], 4)) + '\n'
    #img_cropped = (img_cropped * 255).astype('uint8')
    #temp = Image.fromarray(img_cropped)
    #temp_path = os.path.join(save_dir, item['file_name'][:-9], item['file_name'])
    #print(temp_path)
    #temp.save(temp_path)
with open('E:\\face_cropped\\train_cropped.csv', 'w') as f:
    f.write(train_cropped_out)
'''