import face_recognition
import numpy as np
import cv2

current_id = 0
known_faces = []
unknown_encodings = {}


def recognition_task(face_queue, track_recognition_dict, removed_tracks):
    while True:
        # Make sure there's faces to be processed
        # Important: ONLY REMOVE AND IDENTIFY TRACKS ONCE THE ENTIRE QUEUE OF FRAMES IS PROCESSED.
        if len(face_queue):
            boxes, track_ids, landmarks, frame = face_queue.pop()
            boxes = [(int(top), int(right), int(bottom), int(left)) for (left, top, right, bottom) in boxes]

            fin_encodings = face_recognition.face_encodings(frame, boxes)
            for t_id, encoding in zip(track_ids, fin_encodings):
                if known_faces:
                    distance = face_recognition.face_distance(known_faces, encoding)

                    closest_index = np.argmin(distance)

                    if np.min(distance) < 0.55:
                        closest_index = np.argmin(distance)
                    else:
                        closest_index = 'Unknown'
                        #!!! WARNING: CURRENTLY ONLY SAVES THE EARLIEST ENCODING (when the face is first seen and recognized as "unknown")
                        unknown_encodings[t_id] = encoding
                        #closest_index = len(known_faces)
                        #known_faces.append(encoding)
                else:
                    closest_index = 0
                    known_faces.append(encoding)

                if t_id in track_recognition_dict:
                    track_recognition_dict[t_id].append(closest_index)
            
            print(f'Hello from chicago: {track_ids}, {len(face_queue)}')
            for t in track_recognition_dict.keys():
                print(t, '|' , track_recognition_dict[t])
        else:
            # Only here do you process removed_tracks.
            # This is to ensure that removed tracks are not processed before all the frames in which they were contained have been processed.
            for track_id in removed_tracks:
                # Some tracks are "removed" twice, so this protects against bugs from that.
                if track_id in track_recognition_dict:
                    detections_list = track_recognition_dict[track_id]
                    final_detection = max(detections_list, key=detections_list.count)
                    if final_detection == "Unknown":
                        print(f'track {track_id} is unknown, adding them to known faces as person {len(known_faces)}')
                        known_faces.append(unknown_encodings[track_id])
                    else:
                        print(f'track {track_id} was recognized as person {final_detection}')
                    
                    
                    del track_recognition_dict[track_id]