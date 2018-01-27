# travel_mod
road detect/classification
## 1.模型训练
### 1.每张图片生成一个feature.txt文件,合成一个总的txt.
注意图片的顺序,正负样本的分离,可将原来的txt适当混入
### 2.运行MATLAB,利用svm训练生成新的model.
model文件是怎么生成的?
 ./svm-train -c 1 -g 0.07 -b 1 train0603.txt t.model


-----
遇到的问题:
### 1.TRDetect::extract(int label)或者TRDetect::simpleExtract(const Mat &img, int num)生成向量文档.

### 1.opencv error
```
OpenCV Error: Assertion failed (ssize.area() > 0) in resize, file /home/zc/tools/opencv-2.4.13/modules/imgproc/src/imgwarp.cpp, line 1968
terminate called after throwing an instance of 'cv::Exception'
  what():  /home/zc/tools/opencv-2.4.13/modules/imgproc/src/imgwarp.cpp:1968: error: (-215) ssize.area() > 0 in function resize
```
出错原因:图片没读进来.改了格式,名称.
https://stackoverflow.com/questions/31996367/opencv-resize-fails-on-large-image-with-error-215-ssize-area-0-in-funct
### 2.gdb 
gdb filename
run
bt
### 3.c++中使用c
.c 改为 .cpp
添加 .h文件,其中加入 #pragma once
```
//Automobile.h
#ifndef _AUTOMOBILE_H
#define _AUTOMOBILE_H
#else
//...
#endif 
```
vs
```
//Automobile.h
#pragma once
//... 
```
### 4.static
重复定义变量,前面+static

### 5.#include < string >
错因:string两边没有空格,应为:#include <string>

### 6.'uint32' was not declared in this scope
Under Linux, one normally uses "uint32_t".

### 7.cmake 构建
cmake -->makefile
构建:生成 '.o' 文件

### 8.文件路径的快速复制
点击文件夹,右键复制,不需要打开包含很多内容(图片)的文件夹.

### 9.学会找error
如error:can't find ...,要能快速定位到这里

### 10.QT调试
修改程序后,可清除后重新构建(保险).或者,直接构建即可(快).
删除或添加文件后,需要 右键项目名,然后执行CMake

### 11.程序执行
Command line arguments:
注意模型文件要与可执行文件放在同一目录下
### 12.修改编码
enca -L zh_CN -x utf-8 main.cpp
### 13.传参\函数构建后在子函数调用

