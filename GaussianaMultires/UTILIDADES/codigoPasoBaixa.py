import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2,fftn,ifftn

def pasoBaixa(sigmaF,nscale,imaxe,mult):

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

    radial=[]
    convolved=[]

    for ss in range(nscale):

        f0=0.1
        NsigmaF=sigmaF*mult**ss
        print("Para a escala {}, sigmaF {}".format(ss,NsigmaF))
        print("Cociente: {}".format(NsigmaF/f0))
        num=-np.log(radio/f0)
        denom=2*(np.log(NsigmaF/f0))**2
        component=np.exp(num**2/denom)

        component[0,0]=0.0
        convolved_result = ifft2(IM * component)
        convolved.append(convolved_result)
        realPart = np.real(convolved_result)
        
        BW=2*np.sqrt(2/np.log(2))*np.abs(np.log(NsigmaF/f0))
        print("Ancho de banda na escala {}: {}".format(ss,BW))

        plt.subplot(1, nscale, ss + 1)
        plt.imshow(realPart,cmap="gray")
        plt.title(f'Escala {ss + 1}')

    plt.show()

    plt.imshow(np.sum(np.real(convolved),axis=0),cmap='gray')
    plt.title("reconstrucion")
    plt.show()

def pasoBaixaVideo(sigmaF, nscale, minlonguraOnda, frameArray, mult):
    
    '''aplica o filtrado paso baixa a un video'''
    print(frameArray.shape)
    filas,columnas,frames=frameArray.shape
	
    IM=fftn(frameArray)
    resultados=[]
    if(columnas%2==0):
        xvals=np.arange(-(columnas-1)/2.0,((columnas-1)/2.0)+1)/float(columnas-1)
        
    else:
        xvals=np.arange(-columnas/2.0,columnas/2.0)/float(columnas)
        
    if (filas%2):
        yvals=np.arange(-(filas-1)/2.0,((filas-1)/2.0)+1)/float(filas-1)
        
    else:
        yvals=np.arange(-filas/2.0,filas/2.0)/float(filas)

    if(frames%2):
        zvals=np.arange(-(frames-1)/2.0,((frames-1)/2.0)+1)/float(frames-1)

    else:
        zvals=np.arange(-frames/2.0,frames/2.0)/float(frames)

    x,y,z=np.meshgrid(xvals,yvals,zvals)
    radio=np.sqrt(x**2+y**2+z**2)
    radio=ifftshift(radio)
    radio[0,0]=1.0

    for ss in range(1):
        f0=0.1
        NsigmaF=sigmaF*mult**ss
        print("Para a escala {}, sigmaF {}".format(ss,NsigmaF))
        print("Cociente: {}".format(NsigmaF/f0))
        num=-np.log(radio/f0)
        denom=2*(np.log(NsigmaF/f0))**2
        component=np.exp(num**2/denom)

        component[0,0]=0.0

        resultados.append(ifftn(IM*component))

    for convolution in resultados:
        for fotograma in convolution:
            cv2.imshow('video',np.real(fotograma))
        
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

imaxe=cv2.imread("../DATA/lenna.png",0)

cap=cv2.VideoCapture("../DATA/SCE_bur_11_girando.avi")
frames=[]

while cap.isOpened():

    ret,frame=cap.read()

    if not ret:
        break

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frames.append(frame)

frames=np.array(frames)
#pasoBaixa(0.5,5,imaxe,1/8,2)
pasoBaixaVideo(0.35,3,3,frames,0.8)
