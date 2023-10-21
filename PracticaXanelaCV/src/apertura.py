import cv2
import time
import numpy as np 
from matplotlib import pyplot as plt 

puntosOrixe = []
puntosDestino=[[0,0],[500,0],[500,255],[0,255]]

def onClick(event, x, y, flags, param):
    global puntosOrixe

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(puntosOrixe) < 4:
            puntosOrixe.append([x,y])
            cv2.circle(canvas, (x, y), 5, (0, 255, 0), 2)
            cv2.imshow("primeiro frame", canvas)  # Actualizar la ventana

source = "../DATA/proba.mp4"
cap = cv2.VideoCapture(source)

cv2.namedWindow("primeiro frame")
cv2.setMouseCallback("primeiro frame", onClick)

ret, primeiro_frame = cap.read()
canvas=primeiro_frame.copy()
cv2.imshow("primeiro frame", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

puntosOrixe=np.array(puntosOrixe,dtype=np.int32)
puntosDestino=np.array(puntosDestino,dtype=np.int32)

#aplicamos a homografía a xanela
h,status=cv2.findHomography(puntosOrixe,puntosDestino)

Kernel=(9,9)
k1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,Kernel)

#reproducimos o video coa homografia aplicada
cv2.namedWindow("xanela granxa")
while True:
    ret,frame=cap.read()
    
    if not ret:
        print("Non hai frame")

    imaxeDestino=cv2.warpPerspective(frame,h,(500,255))
    l,a,b=cv2.split(cv2.cvtColor(imaxeDestino,cv2.COLOR_BGR2Lab))
    mediaLuz=np.mean(l)

    #convertindo o frame a Lab e facendo a media de luminosidade podemos saber se e de dia ou de noite
    #sabendo a media de luz, podemos resaltar brancos ou negros para manter o borde da venta sempre diferenciable

    #se hai moita luz, aplicamos erosion
    if mediaLuz>100:
        frameCorrixido=cv2.erode(imaxeDestino,k1)

    else:
        frameCorrixido=cv2.dilate(imaxeDestino,k1)
    
    #aplicamos un suavizado gaussiano o frame
    frameCorrixido=cv2.cvtColor(frameCorrixido,cv2.COLOR_BGR2GRAY)
    frameGaussiano=cv2.GaussianBlur(frameCorrixido,(3,3),0,0)
    bordeXanela=cv2.Laplacian(frameGaussiano,cv2.CV_32F, ksize = 3,scale = 1, delta = 0)

    cv2.imshow("xanela granxa",bordeXanela)

    if cv2.waitKey(1000//60)==113:
        print("rematando sesion")
        break

cap.release()
cv2.destroyAllWindows()