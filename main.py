
import cv2
import numpy as np

# 1. Load image
image = cv2.imread("data/rice.jpg")
image = cv2.resize(image, (1020, 768))

if image is None:
    raise FileNotFoundError("Could not load data/rice.jpg")

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Create a black-and-white mask
_, mask = cv2.threshold(
    gray, 100, 255, cv2.THRESH_BINARY
)

# 4. Clean the mask
kernel = np.ones((3, 3), np.uint8)

clean_mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN, kernel
)

# 5. Find contours
contours, _ = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 6. Draw detected contours
result = image.copy()

cv2.drawContours(
    result, contours, -1, (0, 255, 0), 2
)

# 7. Display results
print("Total contours detected:", len(contours))

cv2.imshow("Original", image)
cv2.imshow("Original Mask", mask)
cv2.imshow("Clean Mask", clean_mask)
cv2.imshow("Detected Rice Grains", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
