import cv2
from RetinaClass import RetinaFace
from time import time as systime

# Open the default camera
cam = cv2.VideoCapture(0)

width, height = 1920, 1080

# Set desired resolution (e.g., 1280x720 for HD)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
#cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


face_detector = RetinaFace()

while True:
    st_time = systime()

    ret, frame = cam.read()

    boxes, conf, land = face_detector.run_inference(frame)

    print(boxes)
    print(conf)
    print(land)
    print("__________")

    
    for (left, top, right, bottom) in boxes:
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 3)

    end_time = systime()

    fps = round(1/(end_time - st_time), 2)
    cv2.putText(frame, f"FPS: {fps}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

    # Display the captured frame
    #cv2.imshow('Camera', frame)
    cv2.imshow('Camera', cv2.resize(frame, dsize=None, fx=0.25, fy=0.25))

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()