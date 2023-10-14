#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt
import numpy as np
import cv2 

def imshow(ventana,image):
    cv2.imshow(ventana,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

window='venta'

imaxe=cv2.imread("flowers.png",-1)
imshow(window,imaxe)

backupImaxe=imaxe.copy()
red=(0,0,255)
cv2.rectangle(imaxe,(0,30),(25,55),red,-1)

imaxe=backupImaxe
imaxe=cv2.cvtColor(imaxe,cv2.COLOR_BGR2GRAY)
imshow(window,imaxe)

media=np.mean(imaxe)
std=np.std(imaxe)
maximo=np.max(imaxe)
minimo=np.min(imaxe)

imaxe=np.array(imaxe,dtype=np.float64)
newImaxe=imaxe-media

newImaxe=np.clip(newImaxe,0,255)
newImaxe=np.uint8(newImaxe)

cv2.resize(newImaxe,(256,256))
cv2.imwrite('imaxeOscurecida.jpg',newImaxe)

imshow(window,newImaxe)