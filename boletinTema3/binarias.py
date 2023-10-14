#!/usr/bin/env python3
import cv2
from matplotlib import pyplot as plt
import numpy as np

def plt_imshow(title,image):
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

rectangle=cv2.imread("rectangle.jpg",-1)
circle=cv2.imread("circle.jpg",-1)
pepper=cv2.imread("pepper.jpeg",-1)

circle=cv2.resize(circle,(pepper.shape[1],pepper.shape[0]))
rectangle=cv2.resize(rectangle,(pepper.shape[1],pepper.shape[0]))

print(circle.shape)
print(pepper.shape)

pepper=cv2.cvtColor(pepper,cv2.COLOR_BGR2RGB)
mascaraCircular=cv2.bitwise_and(circle,pepper,mask=None)
mascaraRectangular=cv2.bitwise_and(rectangle,pepper,mask=None)

bgr=cv2.split(mascaraRectangular)

histB=cv2.calcHist(bgr,[0],None,[256],(0,255),accumulate=False)
histG=cv2.calcHist(bgr,[1],None,[256],(0,255),accumulate=False)
histR=cv2.calcHist(bgr,[2],None,[256],(0,255),accumulate=False)

histW=800
histH=800
binW=int(round(histW/256))
histImage=np.zeros(((histW,histH,3)),dtype=np.uint8)

cv2.normalize(histB,histB,alpha=0,beta=histH,norm_type=cv2.NORM_MINMAX)
cv2.normalize(histG,histG,alpha=0,beta=histH,norm_type=cv2.NORM_MINMAX)
cv2.normalize(histR,histR,alpha=0,beta=histH,norm_type=cv2.NORM_MINMAX)

for i in range(1,256):
    cv2.line(histImage, (int(binW * (i - 1)), histH - int(histB[i - 1][0])),(int(binW * i), histH - int(histB[i][0])), (255, 0, 0), thickness=2)
    cv2.line(histImage, (int(binW * (i - 1)), histH - int(histG[i - 1][0])),(int(binW * i), histH - int(histG[i][0])), (0, 255, 0), thickness=2)
    cv2.line(histImage, (int(binW * (i - 1)), histH - int(histR[i - 1][0])),(int(binW * i), histH - int(histR[i][0])), (0, 0, 255), thickness=2)

#aplicamos las mascaras
plt.subplot(131)
plt.imshow(pepper)
plt.title("pepper")

plt.subplot(132)
plt.imshow(mascaraCircular)
plt.title("mascara circular")

plt.subplot(133)
plt.imshow(mascaraRectangular)
plt.title("mascara rectangular")

plt.show()

#mostramos el histograma
cv2.imshow("histograma",histImage)
cv2.waitKey(0)
cv2.destroyAllWindows()