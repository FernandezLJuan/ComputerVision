#!/usr/bin/env python
import cv2
from matplotlib import pyplot as plt 
import numpy as np 

def plt_imshow(title, image):
    # convertemos a imaxe dende BGR ao espazo de cor RGB e a visualizamos
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap="gray")
    #plt.imshow(image, cmap="jet")
    plt.title(title)
    plt.grid(False)
    plt.show()
    

window='flores'

imaxe=cv2.imread('flowers.png')
imaxe=cv2.cvtColor(imaxe,cv2.COLOR_BGR2Lab)
l,a,b=cv2.split(imaxe)

for red,blue in zip(a,b):
    for i in range(len(red)):
        if red[i]>132:
            
           blue[i]+=50
           red[i]-=20

newImage=cv2.merge((l,a,b))
newImage=cv2.cvtColor(newImage,cv2.COLOR_Lab2BGR)
plt_imshow('a',newImage)