import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--root', type=str, default='/media/han/新增磁碟區/cy/dataset/',help='root path')
args = parser.parse_args()

root_dir = args.root # /media/han/新增磁碟區/cy/dataset/

_is = [1,2,3,10,12,13,22,23]
_js = [6]


training_label_file = open(root_dir+'train_label_new.txt', 'w')
training_file_A = open(root_dir+'train_data_A_new.txt', 'w')
training_file_B = open(root_dir+'train_data_B_new.txt', 'w')

for i in _is:
    for j in _js:
        file_pathes = glob.glob(root_dir+'images/A/'+str(i)+'-'+str(j)+'/*.png')
        for path in file_pathes:
            lab = _is.index(int(i))
            training_label_file.write(str(lab)+'\n')
            training_file_A.write(path+'\n')
            training_file_B.write(root_dir+'images/B/'+str(i)+'-'+str(j)+'/'+path.split('/')[-1]+'\n')
            # print(root_dir+'images/B/'+str(i)+'-'+str(j)+'/'+path.split('/')[-1])
        


training_label_file.close()
training_file_A.close()
training_file_B.close()