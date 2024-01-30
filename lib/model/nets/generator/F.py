import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

def F(x):
    #读取图像
    B,C,H,W=x.shape
    x=x.reshape(H,W)
    #print('x:',x)
    image = x  # 以灰度模式读取图像
#傅里叶变化
    image_np=image.cpu().numpy()
    #print('image_np:',image_np)
    f_transform = np.fft.fft2(image_np)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    #定义低频滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # 计算图像中心
    mask = np.ones((rows, cols), np.uint8)
    r = 3  # 设置半径，可根据需要调整
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0  # 中心区域设为0
    #应用滤波器
    f_transform_shifted_filtered = f_transform_shifted * mask
    #逆傅里叶变化
    f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
    image_filtered = np.fft.ifft2(f_transform_filtered)
    image_filtered = np.abs(image_filtered)
    #print('image_filtered:',image_filtered)
    #显示原图和过滤后图像
    # 合并图像
    #image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    alpha = 0.58  # 权重
    result_np = cv2.addWeighted(image_np.astype(np.float32), alpha, image_filtered.astype(np.float32), 1 - alpha, 0, dtype=cv2.CV_64F)
    
    result=torch.from_numpy(result_np).float()
    result=result.reshape(B,C,H,W)
    #print('result:',result)
    return result
'''

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image_filtered, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
'''

    


'''
# 显示或保存结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

