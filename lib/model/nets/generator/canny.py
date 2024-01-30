import cv2
import numpy as np
from matplotlib import pyplot as plt  # 用于显示图像（可选）

# 读取图像
image = cv2.imread('E:/PatRecon8.30-Thousands data/exp/data/2D_projection_0.jpg')

# 如果要在Matplotlib中显示图像，你可以使用以下代码
#plt.imshow(image, cmap='gray')
#plt.title('Original Image')
#plt.show()

# 对图像应用Canny边缘检测
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, threshold1=30, threshold2=70)  # 调整阈值以获得最佳效果


#result=cv2.addWeighted(gray_image,1,np.zeros_like(gray_image),0,0)
result_with_edges=cv2.addWeighted(gray_image,1,edges,1,0)
result_with_edges=cv2.cvtColor(result_with_edges,cv2.COLOR_GRAY2BGR)
# 显示Canny边缘检测结果
plt.imshow(result_with_edges)
plt.title('Canny Edges')
plt.show()

# 如果你想将边缘图像保存到文件，你可以使用以下代码
cv2.imwrite('E:/PatRecon8.30-Thousands data/exp/data/2D_projection_edges.jpg', result_with_edges)
