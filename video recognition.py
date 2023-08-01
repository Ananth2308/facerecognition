import pickle
import cv2

from predictors import face_rects
from predictors import face_encodings
from predictors import nb_of_matches

# load the encodings + names dictionary
with open("features.pickle", "rb") as f:
    name_encodings_dict = pickle.load(f)

video_cap = cv2.VideoCapture(0)

while True:
    _, frame = video_cap.read()

    encodings = face_encodings(frame)

    names = []

    # loop over the encodings
    for encoding in encodings:

        counts = {}

        for (name, encodings) in name_encodings_dict.items():

            counts[name] = nb_of_matches(encodings, encoding)

        if all(count == 0 for count in counts.values()):
            name = "Unknown"

        else:
            name = max(counts, key=counts.get)


        names.append(name)


    for rect, name in zip(face_rects(frame), names):

        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break


video_cap.release(0)
cv2.destroyAllWindows()