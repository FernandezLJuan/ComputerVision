#!/usr/bin/env python3
import cv2
from matplotlib import pyplot as plt
import numpy as np

imaxe=cv2.imread('flowers.png')

lowContrast=(imaxe*0.5).astype(np.uint8)

highContrast=imaxe*2
highContrast=np.clip(highContrast,0,255).astype(np.uint8)

imaxe=cv2.cvtColor(imaxe,cv2.COLOR_BGR2RGB)
lowContrast=cv2.cvtColor(lowContrast,cv2.COLOR_BGR2RGB)
highContrast=cv2.cvtColor(highContrast,cv2.COLOR_BGR2RGB)

plt.subplot(131)
plt.imshow(imaxe)
plt.title('imagen normal')

plt.subplot(132)
plt.imshow(lowContrast)
plt.title('bajo contraste')

plt.subplot(133)
plt.imshow(highContrast)
plt.title('alto cotraste')

plt.show()