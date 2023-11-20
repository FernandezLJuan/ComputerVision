import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

def logGaussiana(f, f0, sigma):
    return np.exp(-0.5 * ((np.log(f / f0)) / sigma)**2)

def banco_filtros_multiescala(fmax, d, s, num_filtros, f):
    filtros = []
    for i in range(num_filtros):
        f0 = fmax / (d**i)
        filtro = logGaussiana(f, f0, s)
        filtros.append(filtro)
    return filtros

# Parámetros
fmax = 1000  # Frecuencia máxima del primer filtro
d = 2  # Distancia en octavas
s = 0.7  # Escala del filtro
num_filtros = 5  # Número de filtros
f = np.logspace(0, 3, 500)  # Vector de frecuencias

# Crear el banco de filtros
filtros = banco_filtros_multiescala(fmax, d, s, num_filtros, f)

# Cargar la imagen
imagen = cv2.imread('../DATA/lenna.png', cv2.IMREAD_GRAYSCALE)

# Aplicar cada filtro al canal de intensidad de la imagen
imagenes_filtradas = []
for filtro in filtros:
    # Crear un filtro 2D replicando el filtro 1D
    filtro_2d = np.outer(filtro, filtro)
    
    # Realizar la Transformada de Fourier de la imagen y del filtro
    imagen_fft = fft2(imagen)
    filtro_fft = fft2(filtro_2d,shape=imagen.shape)
    
    # Realizar la multiplicación en el dominio de la frecuencia
    imagen_filtrada_fft = imagen_fft * filtro_fft
    
    # Realizar la Transformada inversa de Fourier para volver al dominio del tiempo (espacio)
    imagen_filtrada = np.abs(ifft2(imagen_filtrada_fft))
    
    imagenes_filtradas.append(imagen_filtrada)

# Mostrar las imágenes filtradas
for i, imagen_filtrada in enumerate(imagenes_filtradas):
    cv2.imshow(f'Imagen filtrada {i+1}', imagen_filtrada)
cv2.waitKey(0)
cv2.destroyAllWindows()
