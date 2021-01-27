import glob
import os
from PIL import Image

root_dir = '/media/han/新增磁碟區/cy/dataset/'


file_pathes = []
train_label = []
training_label_file = open(root_dir+'val_labels.txt', 'w')
training_file = open(root_dir+'val_images.txt', 'r')
lines = training_file.readlines()
count = 0

for line in lines:
    # transform to png
    line = line.split('\n')[0]
    # print(line)
    img = Image.open(root_dir+line)
    img = img.convert('L')
    new_path = os.path.join('labels',line.split('/')[2],line.split('/')[3])
    # print(new_path)
    # exit(-1)
    img.save(root_dir+new_path)
    training_label_file.write(new_path+'\n')
    print(count)
    count+=1


training_label_file.close()
training_file.close()


