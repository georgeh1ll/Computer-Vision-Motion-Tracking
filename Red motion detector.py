from __future__ import print_function 

import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt



def main():
    """
    Main method of the program.
    """
 
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    
 
    # Create the background subtractor object
    # Use the last 700 video frames to build the background
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, 
        varThreshold=50, detectShadows=True)
 
    # Create kernel for morphological operation
    # You can tweak the dimensions of the kernel
    # e.g. instead of 20,20 you can try 30,30.
    kernel = np.ones((20,20),np.uint8)
 
    while(1):
 
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        ret, frame = cap.read()
        
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_colour = np.array([170,120,10])
        upper_colour = np.array([180,255,255])
        mask=cv2.inRange(hsv,lower_colour,upper_colour)
        # Use every frame to calculate the foreground mask and update
        # the background
        fg_mask = back_sub.apply(frame)
 
        # Close dark gaps in foreground object using closing
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
 
        # Remove salt and pepper noise with a median filter
        fg_mask = cv2.medianBlur(fg_mask, 5) 
         
        # Threshold the image to make it either black or white
        _, fg_mask = cv2.threshold(fg_mask,200,255,cv2.THRESH_BINARY)
 
        # Find the index of the largest contour and draw bounding box
        fg_mask_bb = fg_mask
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
 
        # If there are no countours
        if len(areas) < 1 :
 
           
            
            # If "q" is pressed on the keyboard, 
            # exit this loop
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
 
            # Go to the top of the while loop
            continue
 
        else:
            # Find the largest moving object in the image
            max_index = np.argmax(areas)

            
            
        # Draw the bounding box
            cnt = contours[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
        
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)

 
        # Draw circle in the center of the bounding box
            x2 = x + int(w/2)
            y2 = y + int(h/2)
     
            cv2.circle(frame,(x2,y2),5,(0,255,0),5)
            
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(0,0,255),2)    
            
          
        # Print the centroid coordinates (we'll use the center of the
        # bounding box) on the image
            text = "x: " + str(x2) + ", y: " + str(y2)
        
            print('Centre of contour box:',text)


        
            cv2.putText(frame, text, (x2 - 5, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
        
        
            
        res=cv2.bitwise_and(frame,frame,mask=mask)


        #Display the resulting frame
        
        masked_img= cv2.imshow('Mask window',res)
        camera_img = cv2.imshow('Camera window',frame)
        
        
        # If "q" is pressed on the keyboard, 
        # exit this loop
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        text_file= open("Coordinate_output.txt", "w")
        text_file.write(text)
    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    print(__doc__)
    main()