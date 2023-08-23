import pickle
from threading import Thread
import time
import cv2

from predictors import face_rects
from predictors import face_encodings
from predictors import nb_of_matches

# load the encodings + names dictionary
with open("features.pickle_test", "rb") as f:
    name_encodings_dict = pickle.load(f)
def name_detect(frame):
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
        print("name is :",name)
        return names