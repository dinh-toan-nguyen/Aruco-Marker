
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import csv
import time

#position of camera in the world coordinate
X_cam = 0 # mm
Y_cam = 0
Z_cam = 0

cap = cv2.VideoCapture(1)

# install resolution for camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

markerLength = 92.9   # Here, our measurement unit is mm.

#--- Get the camera calibration path
calib_path  = ""
mtx = np.loadtxt(calib_path+'cameraMatrix_webcam_B.txt', delimiter=',')
dist = np.loadtxt(calib_path+'cameraDistortion_webcam_B.txt', delimiter=',')

print("----------------------------")
print("camera matrix: ",  mtx)
print("----------------------------")
print("camera Distortion: ", dist)
print("----------------------------")
###------------------ ARUCO TRACKER ---------------------------
Poss_list = []
with open("Data.csv", "w", newline = "") as f:
    wc = csv.writer(f, delimiter= ' ')    
    while (True):
        #start = time.time()
        ret, frame = cap.read()
        
        # operations on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 7
    
        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        # check if the ids list is not empty
        # if no check is added the code will crash
        
        
        if np.all(ids != None):
    
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
            
            # tvec is position of marker's center in camera coordinate
            #(rvec-tvec).any() # get rid of that nasty numpy value array error
    
            # calculate marker's center in world coordinate
            X_world =  X_cam + tvec[0][0][0]
            Y_world =  Y_cam + tvec[0][0][2]
            Z_world =  Z_cam - tvec[0][0][1]
                
            Poss_list.append(X_world)
            # Write to csv file
            wc.writerow([X_world, Y_world, Z_world])
            print(round(X_world,1), round(Y_world,1), round(Z_world,1))
            
            for i in range(0, ids.size):
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 1000)
    
                # show translation vector on the corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str([round(i,1) for i in tvec[i][0]])
                position = tuple(corners[i][0][0])
                cv2.putText(frame, text, position, font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                
            # draw a square around the markers
            aruco.drawDetectedMarkers(frame, corners)
        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
    
        # display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            
            break
        #print(time.time()-start)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# with open("Data.csv", "w", newline = "") as f:
#     wc = csv.writer(f, delimiter= ' ')
#     for x in range(1,5):
#         wc.writerow([X_world, Y_world, Z_world])

# References
# 1. https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
# 2. https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
# 3. https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html