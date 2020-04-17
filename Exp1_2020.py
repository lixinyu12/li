"""
Task [I] - Demonstrating how to compute the histogram of an image using 4 methods.
(1). numpy based
(2). matplotlib based
(3). opencv based
(4). do it myself (DIY)
check the precision, the time-consuming of these four methods and print the result.
"""


import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

###
#please coding here for solving Task [I].
file_name = 'D:\Desktop/lisiyi.jpg'
img   = cv2.imread(file_name)#BGR方式读图
img_gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#颜色转换通道

gray_levels = np.arange(0,256,2)#单独显示红色
N_x = np.zeros_like(gray_levels, dtype=np.float)
#循环灰度等级到256
for (i,level) in enumerate(gray_levels):
    N_x[i] = np.sum(img_gray==level)
plt.plot(gray_levels, N_x)
plt.show()

img =cv2.imread(file_name)
cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#颜色分离通道
index = 121  #画1行四列的图，与 1,2,1 同
plt.subplot(index)
plt.imshow(cimg)
redI   = np.zeros_like(img)
redI[:,:,0]   = img[:,:,0]
red = img[:,:,2]

index += 1
plt.subplot(index)
plt.imshow( redI)

plt.show()








###





"""
Task [II]Refer to the link below to do the gaussian filtering on the input image.
Observe the effect of different @sigma on filtering the same image.
Try to figure out the gaussian kernel which the ndimage has used [Solution to this trial wins bonus].
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
"""

###
#please coding here for solving Task[II]


from scipy.ndimage import filters
im = plt.imread('D:\Desktop/lisiyi.jpg')
index = 141  #画1行四列的图，与 1,4,1 同
plt.subplot(index)
plt.imshow(im)
for sigma in (1, 8, 20):
    im_blur = np.zeros(im.shape, dtype=np.uint8)
    for i in range(3):  #对图像的每一个通道都应用高斯滤波
        im_blur[:,:,i] = filters.gaussian_filter(im[:,:,i], sigma)

    index += 1
    plt.subplot(index)
    plt.imshow(im_blur)
    print(sigma)

plt.show()







"""
Task [III] Check the following link to accomplish the generating of random images.
Measure the histogram of the generated image and compare it to the according gaussian curve
in the same figure.
"""

###
#please coding here for solving Task[III]

mean = (2,2,3)
cov =np.eye(3)#使用numpy.eye生成对角矩阵
x= np.random.multivariate_normal(mean,cov,(200,200))
plt.hist(x.ravel(),bins=256,color='r')#绘制直方图
plt.show()


