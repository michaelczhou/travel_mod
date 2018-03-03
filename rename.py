import os
import sys

def rename():
    count = 1
    #path = sys.path[0]            #得到文件路径
    #path = "/home/zc/Downloads/datesets/data2018/g"
    path = "/home/zc/project/travel_mod-build/g/"
    #path = "/home/zc/project/travel_mod-build/label/commit/"
    filelist = os.listdir(path)   #得到文件名字
    print(filelist)
    for files in filelist:
        oldpath = os.path.join(path, files)  # 原来的文件路径
        #print(oldpath)
        if os.path.isdir(oldpath):           #如果是文件夹则跳过
            continue
        #filename = os.path.splitext(files)[0]  # 文件名
        filetype = os.path.splitext(files)[1]  # 文件扩展名
        #print(filename,"in  ",filetype)
        newpath = os.path.join(path,"{:0>4d}".format(count)+filetype)
        # if not os.path.isfile(newpath)
        os.rename(oldpath, newpath)           #重命名
        print(newpath, 'ok')
        count += 1
rename()