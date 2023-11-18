#Importamos as librerias precisas
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2,fftn,ifftn

def pasoBaixa(sigmaF,nscale,minlonguraOnda,imaxe,mult):

    rows,cols=imaxe.shape

    IM=fft2(imaxe)

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x,y=np.meshgrid(xvals,yvals,sparse=True)
    radio=np.sqrt(x**2+y**2)
    radio = ifftshift(radio)
    radio[0,0]=1.0

    denom=2*np.log(sigmaF)**2
    radial=[]
    convolved=[]

    for ss in range(nscale):

        longuraOnda=minlonguraOnda*mult**ss

        f0=1/longuraOnda
        num=np.log(radio/f0)
        component=np.exp(-(num)/denom)

        component[0,0]=0.0
        convolved_result = ifft2(IM * component)
        convolved.append(convolved_result)
        realPart = np.real(convolved_result)
        
        BW=2*np.sqrt(2/np.log(2))*np.abs(np.log(sigmaF/f0))
        print("Ancho de banda:",BW)

        plt.subplot(1, nscale, ss + 1)
        plt.imshow(fftshift(component), cmap='gray')
        plt.title(f'Escala {ss + 1}')

    plt.show()

    plt.imshow(np.sum(np.real(convolved),axis=0),cmap='gray')
    plt.show()
def pasoBanda(sigmaF,nscale,minlonguraOnda,imaxe,mult,fMax,d):
    #TODO create band pass filter
    rows,cols=imaxe.shape

    IM=fft2(imaxe)

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x,y=np.meshgrid(xvals,yvals,sparse=True)
    radio=np.sqrt(x**2+y**2)
    radio = ifftshift(radio)
    radio[0,0]=1.0

    denom=2*np.log(sigmaF)**2
    radial=[]
    convolved=[]

    for ss in range(nscale):

        f0=fMax/(d**ss)
        num=np.log(radio/f0)
        component=np.exp(-(num)/denom)

        component[0,0]=0.0
        radial.append(fftshift(component))
        convolved_result = ifft2(IM * component)
        convolved.append(convolved_result)
        realPart = np.real(ifft2(component))
        
        BW=2*np.sqrt(2/np.log(2))*np.abs(np.log(sigmaF/f0))
        print("Centro na escala {}: {}".format(ss,f0))

        plt.subplot(1, nscale, ss + 1)
        plt.imshow(fftshift(component),cmap="gray")
        plt.title(f'Escala {ss + 1}')

    plt.show()
    
imaxe=cv2.imread("../DATA/apple.png",0)
# source = "../DATA/SCE_bur_11_girando.avi"
# cap = cv2.VideoCapture(source)
# videoArray=[]

# if not cap.isOpened():
#     print("Erro abrindo o video")

# while True:
#     ret, frame = cap.read() 

#     if not ret:
#         print("Non hai frame")
#         break

#     frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     videoArray.append(frame)

#     cv2.imshow("frame",frame)
#     if cv2.waitKey(1000 // 60) == 113:
#         print("rematando sesión")
#         break

# transform=fftn(videoArray)
# transform=np.array(transform)

# for element in transform:
#     cv2.imshow("transformacion",np.real(element))
#     if cv2.waitKey(1000 // 60) == 113:
#         print("rematando sesión")
#         break

print("Video transformado")
pasoBaixa(2,3,3,imaxe,2.1)
#pasoBanda(0.55,3,3,imaxe,3,1/8,2)