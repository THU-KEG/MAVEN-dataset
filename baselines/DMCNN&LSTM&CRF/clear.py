import os
import shutil

del_list = os.listdir("data")
for f in del_list:
    file_path = os.path.join("data", f)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
