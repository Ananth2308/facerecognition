import pickle
import cv2
import os

from predictors import get_image_paths
from predictors import face_encodings

root_dir = "C:/Users/dell/Desktop/cam pics"
class_names = os.listdir(root_dir)

image_paths = get_image_paths(root_dir, class_names)

name_encodings_dict = {}

nb_current_image = 1

for image_path in image_paths:
    print(f"Image processed {nb_current_image}/{len(image_paths)}")

    image = cv2.imread(image_path)

    encodings = face_encodings(image)

    name = image_path.split(os.path.sep)[-2]

    e = name_encodings_dict.get(name, [])

    e.extend(encodings)

    name_encodings_dict[name] = e
    nb_current_image += 1

with open("features.pickle", "wb") as f:
    pickle.dump(name_encodings_dict, f)