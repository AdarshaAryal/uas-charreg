import shutil
import os
'''
file_names = os.listdir(source_dir)
print(file_names)

for name in file_names:
    for file_name in name:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
'''
source_dir = './Img/'
target_dir = './natural_font/'
for root,subdirs,files in os.walk(source_dir):
    for file in files:
        path = os.path.join(root,file)
        shutil.move(path,target_dir)
