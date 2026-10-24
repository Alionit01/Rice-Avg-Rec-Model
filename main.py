import csv
import numpy as np

image_path = "rice.jpg"
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Could not load rice.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()