# Transforamda de Fourier dunha gaussiana no espazo 3D
# output = exp(-2*pi*pi* (u.^2 *sigma1 + v.^2 *sigma2 + w.^2 * sigma3));

from pyfftw.interfaces.scipy_fftpack import fft2, ifft2, fftshift
import numpy as np
import cv2

def fspecial_gauss2D(size, sigma):

    """Funcion para achar unha funcion gaussiana 2D en Fourier
    """
    rows, cols = size
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                       np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                       sparse = True) #frecuencias espaciais da gaussiana
    g = np.exp(-2.0*np.pi**2*((u**2)*sigma[0] + (v**2)*sigma[1])) #gaussiana no espazo de fourier
    return g


#Lemos a imaxe en gris
imaxe = cv2.imread('../DATA/flower.jpg',0)
if imaxe is None:
    print('Non atopo a imaxe')
    exit(1)
print(imaxe.shape)
#Visualizacion kernel gaussiano na frecuencia
print("Kernel gaussiano")
kernel = fspecial_gauss2D((imaxe.shape[0],imaxe.shape[1]), (50,30))
print(kernel.shape)
cv2.imshow('kernel',(np.floor(cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX))).astype(np.uint8))

#Visualizacion kernel no espazo
print("Kernel espazo")
kernel_esp= np.real(fftshift(ifft2(fftshift(kernel))))
cv2.imshow('kernelespazo',(np.floor(cv2.normalize(kernel_esp, None, 0, 255, cv2.NORM_MINMAX))).astype(np.uint8))

#Filtrado
print("Filtrando")
result = np.real((ifft2(fftshift(kernel)*fft2(imaxe))))
cv2.imshow('filtrado',(np.floor(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX))).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()