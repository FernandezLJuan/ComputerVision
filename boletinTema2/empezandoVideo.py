#!/usr/bin/env python
# coding: utf-8
import cv2 

source=0 #lemos webcam
window='webcam'

#lemos video con videocapture
cap=cv2.VideoCapture(source)

ancho=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames=int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_avi = cv2.VideoWriter('salida.avi', fourcc, 20.0, (ancho, alto),isColor=False)

print("Dimensiones: (",ancho,",",alto,")")
print("Fps: ",frames)

while True:

    tieneFrame,frame=cap.read()
    if not tieneFrame:
        break

    grayFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    out_avi.write(grayFrame)

    cv2.imshow(window,grayFrame)

    if cv2.waitKey(25)==113:
        break

cap.release()
out_avi.release()
cv2.destroyAllWindows()