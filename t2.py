import cv2
from RetinaClass import RetinaFace
from time import time as systime
from post_process import RetinaFacePostPostprocessor
from supervision import ByteTrack, Detections
import numpy as np
import face_recognition

#def worker(shared_list):
#    shared_list.append(4)

#with Manager() as manager:
#    # Queue of faces to be processed
#    face_queue = manager.Queue()  # Each item is a packet: (face_img_array, t_id)
#    # IMPORTANT: ADD THE TIMESTAMP OF DETECTION TO THE FACE QUEUE
#    track_recognitions = manager.list([1, 2, 3])
#    p = Process(target=worker, args=(face_queue,))
#    p.start()
#    p.join()
#    print(shared_list) # Output: [1, 2, 3, 4]

# Open the default camera
cam = cv2.VideoCapture(0)

width, height = 1920, 1080

# Set desired resolution (e.g., 1280x720 for HD)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


#face_detector = RetinaFace()
test_facer = RetinaFacePostPostprocessor((width, height), (640, 608))

test_tracker = ByteTrack(frame_rate=20, track_activation_threshold=0.5, minimum_matching_threshold=0.99)

while True:
    st_time = systime()

    ret, frame = cam.read()

    #boxes, conf, land = face_detector.run_inference(frame)

    #print(boxes)
    #print(conf)
    #print(land)
    #print("__________")
   # print('aa', frame.shape)

    locs = face_recognition.face_locations(cv2.cvtColor(cv2.resize(frame, None, fx=0.15, fy=0.15), cv2.COLOR_BGR2RGB))

    boxes = [(left*6.6, top*6.6, right*6.6, bottom*6.6) for (top, right, bottom, left) in locs]
    conf = np.ones(len(boxes))

    print(boxes)

   # print(boxes)q

    dets = Detections(
        xyxy=np.array(boxes) if boxes else np.empty((0,4)),
        confidence=np.array(conf)
    )

    det_results = test_tracker.update_with_detections(dets)

    
    for box, track_id in zip(det_results.xyxy, det_results.tracker_id):
        left, top, right, bottom = map(int, box)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.rectangle(frame, (left, bottom), (right, bottom+50), (0, 0, 255), -1)
        cv2.putText(frame, f"Track: {track_id}", (left, bottom+25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)


    end_time = systime()

    fps = round(1/(end_time - st_time), 2)
    cv2.putText(frame, f"FPS: {fps}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

    # Display the captured frame
    cv2.imshow('Camera', frame)
    #cv2.imshow('Camera', cv2.resize(frame, dsize=(640, 608)))

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()