from glob import glob

import dlib
import os
import cv2
import numpy as np

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("C:/Users/dell/Desktop/models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("C:/Users/dell/Desktop/models/dlib_face_recognition_resnet_model_v1.dat")
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def get_image_paths(root_dir, class_names):
    #grab the paths to the images in our dataset
    image_paths = []

    # loop over the class names
    for class_name in class_names:
        # grab the paths to the files in the current class directory
        class_dir = os.path.sep.join([root_dir, class_name])
        class_file_paths = glob(os.path.sep.join([class_dir, '*.*']))


        for file_path in class_file_paths:

            ext = os.path.splitext(file_path)[1]


            if ext.lower() not in VALID_EXTENSIONS:
                print("Skipping file: {}".format(file_path))
                continue


            image_paths.append(file_path)

    return image_paths


def face_rects(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)
    return rects


def face_landmarks(image):
    return [shape_predictor(image, face_rect) for face_rect in face_rects(image)]


def face_encodings(image):

    return [np.array(face_encoder.compute_face_descriptor(image, face_landmark))
            for face_landmark in face_landmarks(image)]


def nb_of_matches(known_encodings, unknown_encoding):

    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    small_distances = distances <= 0.6
    return sum(small_distances)