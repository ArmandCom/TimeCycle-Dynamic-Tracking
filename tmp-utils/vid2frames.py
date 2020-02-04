import numpy as np
import cv2
name = 'nhl'
cap = cv2.VideoCapture('/data/Armand/TimeCycle/' + name + '.mp4')
i=0
while(i<5000):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    cv2.imwrite('/data/Ponc/tracking/JPEGImages/480p/nhl/'+"{:05d}".format(i)+'.jpeg', frame)
    i+=1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()