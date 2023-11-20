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

    radial=[]
    convolved=[]

    for ss in range(nscale):

        f0=0.1
        NsigmaF=sigmaF*mult**ss
        print("Para a escala {}, sigmaF {}".format(ss,NsigmaF))
        print("Cociente: {}".format(NsigmaF/f0))
        num=-np.log(radio/f0)
        denom=2*(np.log(NsigmaF/f0))**2
        component=np.exp(num/denom)

        component[0,0]=0.0
        convolved_result = ifft2(IM * component)
        convolved.append(convolved_result)
        realPart = np.real(convolved_result)
        
        BW=2*np.sqrt(2/np.log(2))*np.abs(np.log(NsigmaF/f0))
        print("Ancho de banda na escala {}: {}".format(ss,BW))

        # plt.subplot(1, nscale, ss + 1)
        # plt.imshow(realPart,cmap="gray")
        # plt.title(f'Escala {ss + 1}')

    #plt.show()

    plt.imshow(np.sum(np.real(convolved),axis=0),cmap='gray')
    plt.title("reconstrucion")
    plt.show()

def pasoBanda(sigmaF,nscale,imaxe,fMax,d):
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

    resultados=[]

    for ss in range(nscale):

        f0=fMax/d**ss
        NsigmaF=sigmaF*f0 #mantemos sigmaF/f0 constante

        num=np.log(radio/f0)
        denom=2*np.log(NsigmaF/f0)**2

        component=np.exp(-num**2/denom)

        component[0,0]=0.0
        convolved_result = ifft2(IM * component)
        resultados.append(convolved_result)
        
        BW=2*np.sqrt(2/np.log(2))*np.abs(np.log(NsigmaF/f0))
        print("Ancho de banda na escala {}, {}".format(ss,BW))
        print("Centro na escala {}: {}".format(ss,f0))

    # for ss in range(nscale):
    #     plt.subplot((nscale // 5) + 1, 5, ss + 1)
    #     plt.imshow(np.real(resultados[ss]),cmap="gray")
    #     plt.title(f'Escala {ss + 1}')

    #plt.plot((component))
    plt.show()

    plt.imshow(np.sum(np.real(resultados),axis=0),cmap='gray')
    plt.title("reconstrucion")
    plt.show()
    
imaxe=cv2.imread("../DATA/lenna.png",0)

print("Video transformado")
pasoBaixa(0.35,3,3,imaxe,0.8)
#pasoBanda(0.6,5,imaxe,1/8,2)