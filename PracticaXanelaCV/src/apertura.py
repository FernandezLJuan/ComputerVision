import cv2
import time
import numpy as np

puntosOrixe = []
puntosDestino = [[0, 0], [500, 0], [500, 255], [0, 255]]

def onClick(event, x, y, flags, param):
    global puntosOrixe

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(puntosOrixe) < 4:
            puntosOrixe.append([x, y])
            cv2.circle(canvas, (x, y), 5, (0, 255, 0), 2)
            cv2.imshow("primeiro frame", canvas)  # Actualizar la ventana

source = "../DATA/proba.mp4"
cap = cv2.VideoCapture(source)

cv2.namedWindow("primeiro frame")
cv2.setMouseCallback("primeiro frame", onClick)

ret, primeiro_frame = cap.read()
canvas = primeiro_frame.copy()
cv2.imshow("primeiro frame", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

puntosOrixe = np.array(puntosOrixe, dtype=np.int32)
puntosDestino = np.array(puntosDestino, dtype=np.int32)

# Aplicamos la homografía a la ventana
h, status = cv2.findHomography(puntosOrixe, puntosDestino)

Kernel = (5, 5)
k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Kernel)
filtroHorizontal=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

# Reproducimos el video con la homografía aplicada
cv2.namedWindow("canny derivada")
cv2.namedWindow("xanela granxa")
while True:
    ret, frame = cap.read()

    if not ret:
        print("Non hai frame")

    imaxeDestino = cv2.warpPerspective(frame, h, (500, 255))
    l, a, b = cv2.split(cv2.cvtColor(imaxeDestino, cv2.COLOR_BGR2Lab))
    mediaLuz=np.mean(l)
    # Convertindo el frame a Lab e facendo a media de luminosidade podemos saber se e de día ou de noite
    # Sabendo a media de luz, podemos resaltar brancos ou negros para manter o borde da venta sempre diferenciable

    #Se hai moita luz, aplicamos erosion
    if mediaLuz > 100:
        limiar = 60
        frameCorrixido = cv2.erode(imaxeDestino, k1)

    #senon, dilatacion
    else:
        limiar = 15
        frameCorrixido = cv2.dilate(imaxeDestino, k1)

    # Aplicamos un suavizado gaussiano al frame

    frameCorrixido=cv2.GaussianBlur(frameCorrixido,(9,9),2)
    bordeXanela = cv2.Canny(frameCorrixido, threshold1=limiar, threshold2=limiar * 3)
    bordeXanela=cv2.filter2D(bordeXanela,cv2.CV_8U,filtroHorizontal,cv2.BORDER_DEFAULT)
    
    bordesX,bordesY=np.where(bordeXanela!=0)
    
    cv2.imshow("canny derivada", bordeXanela)
    cv2.imshow("xanela granxa", imaxeDestino)
	
    if cv2.waitKey(1000 // 60) == 113:
        print("rematando sesión")
        break

cap.release()
cv2.destroyAllWindows()
