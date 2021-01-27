import glob
import os
import cv2

root_dir = '../dataset/images/B/'

_is = [1,2,3,10,12,13,22,23]
_js = [6]

file_pathes = []
train_label = []
training_label_file = open('../dataset/train_label.txt', 'w')

for i in _is:
    for j in _js:
        file_pathes += glob.glob(root_dir+str(i)+'-'+str(j)+'/*.jpg')
        



training_file = open('../dataset/train_data.txt', 'w')
count = 0
for path in file_pathes:
    try:
        # transform to png
        image = cv2.imread(path)
        cv2.imwrite(path[:-3]+'png' , image)

        training_file.write(path[3:-3]+'png\n')
        i = _is.index(int(path.split('/')[-2].split('-')[0]))
        training_label_file.write(str(i)+'\n')
    except:
        print(path)

# remove jpg file
for path in file_pathes:
    os.remove(path)

training_label_file.close()
training_file.close()


