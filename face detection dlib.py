import dlib
import cv2
img = cv2.imread("C:\\Users\\dell\\Desktop\\picture1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hogFaceDetector = dlib.get_frontal_face_detector()
face = hogFaceDetector(gray, 1)
for (i, rect) in enumerate(face):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 10))
    plt.imshow(img_rgb)
    plt.axis('off')