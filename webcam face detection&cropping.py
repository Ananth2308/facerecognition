import cv2
import os
import dlib
# hogFaceDetector = dlib.get_frontal_face_detector()
# face = hogFaceDetector(gray, 1)
#
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

directory = input("\n ENTER YOUR NAME  ")
parent_dir = "C:/Users/dell/Desktop/cam pics"
path = os.path.join(parent_dir, directory)
if os.path.exists(path):
    pass
else :
    new_path = os.mkdir(path)

# if os.path.exists(path):


print("THANK YOU ", directory)
print('DIRECTORY CREATED SUCCESSFULLY')
print(path)
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    # face = hogFaceDetector(gray_image, 1)
    # for (i, rect) in enumerate(face):
    #     x = rect.left()
    #     y = rect.top()
    #     w = rect.right() - x
    #     h = rect.bottom() - y
    #     cv2.rectangle(video_capture, (x, y), (x + w, y + h), (0, 255, 0), 1)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


i = 0
while True:

    result, video_frame = video_capture.read()
    if result is False:
        break

    faces = detect_bounding_box(video_frame)

    cv2.imshow("My Face Detection Project", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face = video_frame[y:y + h, x:x + w]
                cv2.imshow(f"Cropped Face {i}", face)
                # name = f'main{i}.jpg'
                # cv2.imwrite(name, face)
                i = i + 1
                cv2.imwrite(str(path) + '/' + str(directory) + '.' + str(i) + ".jpg", face)

    if cv2.waitKey(1) & 0xFF == ord("1"):
        break


video_capture.release()
cv2.destroyAllWindows()