import pickle
from threading import Thread
import time
import cv2

from predictors import face_rects
from predictors import face_encodings
from predictors import nb_of_matches

# load the encodings + names dictionary
with open("features.pickle", "rb") as f:
    name_encodings_dict = pickle.load(f)
# defining video capture function
video_cap = cv2.VideoCapture(0)
# looping over the frames in the video capture





while True:
    _, frame = video_cap.read()

    encodings = face_encodings(frame)
    names = []

    # loop over the encodings
    for encoding in encodings:

        counts = {}

        for (name, encodings) in name_encodings_dict.items():

            counts[name] = nb_of_matches(encodings, encoding)
        # checking for the name that has max no. of counts
        if all(count == 0 for count in counts.values()):
            name = "Unknown"

        else:
            name = max(counts, key=counts.get)


        names.append(name)




    # constructing a rectangle over the face in the webcam and the name which had the maximum number of matches
    for rect, name in zip(face_rects(frame), names):

        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # opening the webcam



    cv2.imshow("test_image",frame)
    if cv2.waitKey(1) == ord('q'):
        break



#t1 = Thread(target=vc)
#t1.start()


video_cap.release(0)
cv2.destroyAllWindows()