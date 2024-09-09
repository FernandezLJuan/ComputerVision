import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2,fftn,ifftn

class filtroPasoBanda:

		def __init__(self,sigma,fMax,nscale,d):

			'''Crea o banco de filtros logGaussiana paso banda empregando os parametros: 
        		sigmaF ---- apertura do banco en cada escala, empregado para axustar o ancho de banda
        		nscale ---- numero de escalas do filtro, as escalas baixas filtran frecuencias altas e as altas frecuencias baixas
        		fMax   ---- frecuencia maxima do primeiro filtro do banco, empregase para calcular a frecuencia central do filtro en cada escala
        		d      ---- distancia entre filtros de distintas escalas (en oitavas)'''

			self.sigmaF=sigma
			self.fmax=fMax
			self.nscale=nscale
			self.d=d
			self.f0=0.0
			self.filtros=[]
            
            
		def construirBanco(self,shape):
			'''construe o banco de filtros paso banda coas dimensions dunha imaxe ou video empregando os parametros proporcionados,
			   esta funcion chamase internamente cando se quere aplicar o filtro, os filtros gardanse na lista self.filtros'''
			
			if len(shape)==2:
				filas,columnas=shape
				
				if(columnas%2==0):
					xvals=np.arange(-(columnas-1)/2.0,((columnas-1)/2.0)+1)/float(columnas-1)
					
				else:
					xvals=np.arange(-columnas/2.0,columnas/2.0)/float(columnas)
					
				if (filas%2):
					yvals=np.arange(-(filas-1)/2.0,((filas-1)/2.0)+1)/float(filas-1)
					
				else:
					yvals=np.arange(-filas/2.0,filas/2.0)/float(filas)
					
				x,y=np.meshgrid(xvals,yvals,sparse=True)
				radio=np.sqrt(x**2+y**2)
				radio=ifftshift(radio)
				radio[0,0]=1.0
			
			elif len(shape)==3:
				filas,columnas,frames=shape
				
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
					
				x,y,z=np.meshgrid(xvals,yvals,zvals,sparse=True)
				radio=np.sqrt(x**2+y**2+z**2)
				radio=ifftshift(radio)
				radio[0,0]=1.0
			
			for ss in range(self.nscale):
				self.f0=self.fmax/self.d**ss
				NsigmaF=self.sigmaF*self.f0 #novo valor de sigma, para manter constante o ancho de banda nos filtros (relacion sigmaF/f0)
					
				numerador=np.log(radio/self.f0)
				denominador=2*np.log(NsigmaF/self.f0)**2
				componhente=np.exp(-numerador**2/denominador)
				componhente[0,0]=0.0
				
				BW=2*np.sqrt(2/np.log(2))*np.abs(np.log(NsigmaF/self.f0))
				
				print("Filtro na escala {} con centro en {}".format(ss,self.f0))
				print("Ancho de banda na escala {}: {}".format(ss,BW))
				self.filtros.append(componhente)
				
		def filtrarImaxe(self,imaxe):
		
			'''construe un banco de filtros coas dimensions da imaxe e aplicallo, devolve o resultado da filtraxe, tanto a compoñente real como a complexa'''
			self.construirBanco(imaxe.shape)

			resultados=[]
			IM=fft2(imaxe)      		

			for ss in range(self.nscale):
				resultados.append(ifft2(IM*self.filtros[ss]))
				
			return resultados
		
		def filtrarVideo(self,video):
			'''recibe un argumento de tipo cv2.videoCapture e filtra frame a frame. Neste caso, a propia funcion mostra o resultado da convolucion en cada escala'''
			
			frames=[]

			while video.isOpened():
				ret,frame=video.read()

				if not ret:
					break

				frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
				frames.append(frame)

			frames=np.array(frames)
			self.construirBanco(frames.shape)
			transform=fftn(frames)
			resultados=[]

			#aplicamoslle cada filtro construido o video
			for filtro in self.filtros:
				print("Aplicando filtro o video")
				resultados.append(ifftn(transform*filtro))
			
			#para cada resultado de filtraxe mostramos o video
			for convolution in resultados:
				for fotograma in convolution:
					cv2.imshow('video',np.real(fotograma))
				
					if cv2.waitKey(25) & 0xFF == ord('q'):
						break
					
		def reconstrucion(self,resultados):
			'''metodo que toma os resultados da fitlraxe e reconstrue a imaxe orixinal, para isto suma as compoñentes reais dos resultados das convolucions nas multiples escalas'''
			
			originalImage=np.sum(np.real(resultados),axis=0)
			
			return originalImage

		def fusionImaxe(self,imaxe):

			'''toma os resultados de filtrar a imaxe e fusionaos empregando unha piramide laplaciana'''

			resultados=self.filtrarImaxe(imaxe)
			resultados=np.real(resultados)

			piramideGaussiana=[]
			piramideLaplaciana=[]

			for result in resultados:
				piramideGaussiana.append(cv2.pyrDown(result))

			for i,result in enumerate(resultados[:-1]):

				piramideLaplaciana.append(cv2.subtract(result,cv2.resize(cv2.pyrUp(piramideGaussiana[i+1]),(225,225))))

			piramideLaplaciana.append(cv2.resize(piramideGaussiana[-1],(225,225)))

			imaxeFusionada=np.sum(piramideLaplaciana,axis=0)
			imaxeFusionada=np.clip(imaxeFusionada,0,255).astype(np.uint8)

			cv2.imshow("laplaciana",imaxeFusionada)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		def visualizarFiltros(self):
			'''permite visualizar os filtros no espazo de fourier e no dominio espacial'''
			
			print("Filtros no dominio de fourier")
			for ss in range(self.nscale):
				plt.subplot((self.nscale // 5) + 1, 5, ss + 1)
				plt.imshow(fftshift(self.filtros[ss]),cmap="gray")
				plt.title(f'Escala {ss + 1}')

			plt.show()
			
			print("Filtros no dominio espacial")
			for ss in range(self.nscale):
				plt.subplot((self.nscale // 5) + 1, 5, ss + 1)
				plt.imshow(np.abs(fftshift(ifft2((self.filtros[ss])))),cmap="gray")
				plt.title(f'Escala {ss + 1}')
				
			plt.show()
        	
banco1=filtroPasoBanda(0.5,1/8,5,2)
video=cv2.VideoCapture("../DATA/SCE_bur_11_girando.avi")
banco1.filtrarVideo(video)