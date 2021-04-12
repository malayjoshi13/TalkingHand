import cv2
import os
import sys

mode= sys.argv[1]
label_name = sys.argv[2]
num_samples = int(sys.argv[3])

if not os.path.exists(#paste this code in python idle and save this as malayhand.py/"+mode+"/"+label_name):
    os.makedirs("C:/Users/lenovo/DATA/"+mode+"/"+label_name)
else:
    print("already exists")  

directory = "C:/Users/lenovo/DATA/"+mode+"/"+label_name+"/"

cap = cv2.VideoCapture(0)

start = False
count = 0
subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(2,False,20,True)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        continue

    if count == num_samples:
        break

    cv2.rectangle(frame, (300,65),(635,415), (255, 255, 255), 2)

    if start:
        roi = frame[65:415, 300:635]
        roi = cv2.resize(roi, (64, 64))  
        mask = subtractor.apply(roi, -1)
        cv2.imshow("mask", mask)
        p = (cv2.countNonZero(mask)*100)/(4096)
        if p>5: 
            cv2.imwrite(directory+str(count + 1)+'.jpg', mask)
            count += 1
        else:
            continue
            
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count), (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)
        
    k = cv2.waitKey(10)
    if k == ord('a'):  #start/pause
        start = not start
    if k == ord('q'):  #close
        break
    

cap.release()
cv2.destroyAllWindows()