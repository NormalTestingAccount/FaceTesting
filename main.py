import cv2
from time import time as systime
from post_process import RetinaFacePostPostprocessor
from supervision import ByteTrack, Detections
import numpy as np
from multiprocessing import Process, Manager
from recognition import recognition_task

if __name__ == '__main__':
    # Used to choose how many frames to skip before sending one for Face Recognition
    frame_skip_interval = 10
    current_frame = frame_skip_interval

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

    with Manager() as manager:
        # Queue of faces to be processed
        face_queue = manager.list()  # Each item is a packet: (boxes, track_ids, landmarks, frame)
        # IMPORTANT: ADD THE TIMESTAMP OF DETECTION TO THE FACE QUEUE
        # Track_Id: arr[int] (each entry in array is a matched UUID)
        track_recognition_dict = manager.dict()

        removed_tracks = manager.list()

        p = Process(target=recognition_task, args=(face_queue, track_recognition_dict, removed_tracks))  # Change to "run" in recognition.py
        p.start()

        while True:
            st_time = systime()

            ret, frame = cam.read()

            #boxes, conf, land = face_detector.run_inference(frame)

            #print(boxes)
            #print(conf)
            #print(land)
            #print("__________")
        # print('aa', frame.shape)

            boxes, conf, landmarks = test_facer.process_output(frame)

            #boxes = tuple(map(int, a) for a in )

            #print(boxes)
            #print(landmarks)
            #quit()

        # print(boxes)q

            dets = Detections(
                xyxy=np.array(boxes) if boxes else np.empty((0,4)),
                confidence=np.array(conf),
                data={'landmarks':np.array(landmarks)}
            )

            det_results = test_tracker.update_with_detections(dets)

            for rmvd_track in test_tracker.removed_tracks:
                print(f'track {rmvd_track.external_track_id} is gone permanently')
                if rmvd_track.external_track_id in track_recognition_dict:
                    removed_tracks.append(rmvd_track.external_track_id)

            if det_results.data:

                info_package = (det_results.xyxy, det_results.tracker_id, det_results.data['landmarks'])
                for t_id in det_results.tracker_id:
                    if t_id not in track_recognition_dict:
                        track_recognition_dict[t_id] = manager.list()

                current_frame -= 1
                if current_frame <= 0:
                    current_frame = frame_skip_interval
                    face_queue.append(info_package + (frame,))
                
                for box, track_id, lmark_arr in zip(*info_package):
                    left, top, right, bottom = map(int, box)
                    detected_uuid = track_recognition_dict[track_id] if track_id in track_recognition_dict else "???"

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                    cv2.rectangle(frame, (left, bottom), (right, bottom+50), (0, 0, 255), -1)
                    cv2.putText(frame, f"Track: {track_id}, UUID: {detected_uuid}", (left, bottom+25), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)

                    x_coords = lmark_arr[::2]
                    y_coords = lmark_arr[1::2]
                    for (x, y) in zip(x_coords, y_coords):
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), thickness=-1)


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
        p.terminate()