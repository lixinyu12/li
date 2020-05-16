'''
In this class we will learn Fourier Transform in Python.
本次实验课程学习用Python来做傅立叶变换的应用。
'''
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2

#from scipy.ndimage import gaussian_filter
from scipy import ndimage

###一维曲线的傅立叶变换。观察周期信号在频域的稀疏特性。
'''
1 - Test the symmetrical properties of the magnitude of the Frequency.
'''
'''
香浓采样定理：
离散化表达正弦波（频率为fi，角频率为wi），需要采样。时域的离散，对应频域的周期化。且采样间隔Ts对应频域的周期fp = 1/Ts
因此频域的周期fp必须大于2倍的输入信号频率fi。
注意当输入信号在一个周期内分成N份时，对应的频域周期fp=N。
因此只有当fi<N/2时，FFT变换才不会出现混叠。可以观察fi=1,2,49,50,100，对应的信号图和频谱图。
fi=100时与fi=1时，是相同的。完全重叠。

通过频域变换的公式可以知道，ff[0]对应直流分量，
当i>0时，abs(ff[i]) = abs(ff[N-i]), angle(ff[i])=-angle(fft[i])

因此N个点的时域信号的FFT变换得到的N个频谱，只能对应最高频率为N/2的基函数。否则会有混叠。
'''
# T = 1 # the periodic of the sin. is 1 second
# N = 100
# t = np.linspace(0, T, 100)
# w1 = 2*np.pi/T  # f1 = 1
# w2 = 10*np.pi/T # f2 = 5
# wt = 49*2*np.pi/T # ft = 100 will cause alias
# wt = 100*2*np.pi/T
# #f = np.sin(w1*t) + np.sin(w2*t) + np.sin(wt*t)
# f  = np.sin(wt*t)
# ff = pylab.fft(f, N)
# fm = np.abs(ff)
# fa = np.angle(ff, deg=True)
# # for i in range(1,int(N/2)):
# #     print(fa[i], fa[N-i], np.abs(fa[i]-fa[N-i]))
# fig, axs = plt.subplots(1,3)
# axs[0].plot(t, f)
# axs[0].set_title('signal')
# axs[1].stem(np.abs(ff))
# axs[1].set_title('mag. of freq.')
# axs[2].stem(fa)
# axs[2].set_title('angle of freq.')
# plt.show()

'''
实验任务1- 仿照上面的例子，自己设计一个复杂的周期函数然后求其傅立叶频域的频谱图。
'''
T = 1
N = 100
t = np.linspace(0, T, 100)
m1 = 2*np.pi/T
m2 = 10*np.pi/T
mt = 100*np.pi/T
f = np.cos(m1*t)+np.sin(m2*t)+np.tan(mt*t)
ff = pylab.fft(f, N)
fm = np.abs(ff)
fa = np.angle(ff, deg=True)
fig,axs=plt.subplots(1,3)
axs[0].plot(t, f)
axs[0].set_title('1')
axs[1].stem(fm)
axs[1].set_title('2')
axs[2].stem(fa)
axs[2].set_title('3')
plt.show()









'''
2 - 2d Image Gaussian blur. 观察利用空间卷积和频域乘积对同一幅画面的作用
'''
# from scipy import fftpack

# Here change to your own image's path
# file_name = 'D:\Desktop/canoe.tif'
# img = cv2.imread(file_name, 0 )
# fig, axes = plt.subplots(1, 3)
# axes[0].imshow(img, cmap ='gray')
# axes[0].set_title('source')
#
# img_cv = ndimage.gaussian_filter(img, sigma=2)
# axes[1].imshow(img_cv, cmap='gray')
# axes[1].set_title('convolve gaussian kernel')
#
# #get the gaussian kernel
# w  = 100
# h  = 100
#
# in_mask = np.zeros((h, w), dtype=np.float32)
# in_mask[int(h/2), int(w/2)] = 1
# img_kernel = ndimage.gaussian_filter(in_mask, sigma=2)
#
# kernel = img_kernel[50-6:50+6, 50-6:50+6]
#
# # Padded fourier transform, with the same shape as the image
# # We use :func:`scipy.signal.fftpack.fft2` to have a 2D FFT
# kernel_ft = fftpack.fft2(kernel, shape=img.shape[:2], axes=(0, 1))

##manual pad zeros, same results.
# kernel_pad = np.zeros_like(img,dtype='float')
# # kh, kw = kernel.shape[:2]
# # kernel_pad[:kh, :kw]= kernel
# # kernel_pad_ft = fftpack.fft2(kernel_pad,  axes=(0,1))
# # fig1,axs = plt.subplots(1,3)
# # axs[0].imshow(np.abs(pylab.fftshift(kernel_ft)))
# # axs[1].imshow(np.abs(pylab.fftshift(kernel_pad_ft)))
# # axs[2].imshow(np.abs(kernel_pad_ft-kernel_ft))

# convolve
# img_ft = fftpack.fft2(img, axes=(0, 1))
# img2_ft = kernel_ft * img_ft
# img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real
# # clip values to range
# img2 = np.clip(img2, 0, 1)
#
# axes[2].imshow(img, cmap='gray')
# axes[2].set_title('blur via Gaussian filter')
# plt.show()
# #
#
'''
实验任务2- 仿照上面高斯滤波的例子，尝试自己编写 ndimage.gaussian_laplace 滤波器与卷积的对比。
'''
from scipy import fftpack
file_name = 'D:\Desktop/canoe.tif'
img = cv2.imread(file_name,0 )
fig, axes = plt.subplots(1, 3)
axes[0].imshow(img, cmap ='gray')
axes[0].set_title('source')

img_cv = ndimage.gaussian_filter(img, sigma=2)
axes[1].imshow(img_cv, cmap='gray')
axes[1].set_title('convolve gaussian kernel')

#get the gaussian kernel
w  = 100
h  = 100

in_mask = np.zeros((h, w), dtype=np.float64)
in_mask[int(h/2), int(w/2)] = 1
img_kernel = ndimage.gaussian_filter(in_mask, sigma=2)

kernel = img_kernel[50-2:50+2, 50-2:50+2]
kernel_ft = fftpack.fft2(kernel, shape=img.shape[:2], axes=(0, 1))

img_ft = fftpack.fft2(img, axes=(0, 1))
img2_ft = kernel_ft * img_ft
img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real
img2 = np.clip(img2, 0, 1)

axes[2].imshow(img, cmap='gray')
axes[2].set_title('blur via Gaussian filter')
plt.show()


kernel_pad = np.zeros_like(img,dtype='float')
kh, kw = kernel.shape[:2]
kernel_pad[:kh, :kw]= kernel
kernel_pad_ft = fftpack.fft2(kernel_pad,  axes=(0,1))
fig1,axs = plt.subplots(1,3)
axs[0].imshow(np.abs(pylab.fftshift(kernel_ft)))
axs[1].imshow(np.abs(pylab.fftshift(kernel_pad_ft)))
axs[2].imshow(np.abs(kernel_pad_ft-kernel_ft))
plt.show()






