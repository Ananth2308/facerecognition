import pickle
import cv2

from predictors import face_rects
from predictors import face_encodings
from predictors import nb_of_matches

# load the encodings + names dictionary
with open("features.pickle", "rb") as f:
    name_encodings_dict = pickle.load(f)


image = cv2.imread("C:\\Users\\dell\\Pictures\\Camera Roll\\WIN_20230723_02_17_51_Pro.jpg")

encodings = face_encodings(image)

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


for rect, name in zip(face_rects(image), names):

    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.imshow("image", image)
cv2.waitKey(0)