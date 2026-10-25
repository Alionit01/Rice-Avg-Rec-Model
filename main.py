
import cv2
import numpy as np

image = cv2.imread("data/rice.jpg")
image = cv2.resize(image, (1020, 768))

if image is None:
    raise FileNotFoundError("Could not load data/rice.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(
    gray, 100, 255, cv2.THRESH_BINARY
)

kernel = np.ones((3, 3), np.uint8)
clean_mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN, kernel
)

contours, _ = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

min_area = 100
rice_contours = [
    contour for contour in contours
    if cv2.contourArea(contour) >= min_area
]

result = image.copy()

for i, contour in enumerate(rice_contours, start=1):
    # Find a rotated rectangle around the grain
    rect = cv2.minAreaRect(contour)

    # Get its two side lengths in pixels
    width, height = rect[1]

    # The longer side is our estimated grain length
    length_px = max(width, height)

    # Draw the rotated rectangle
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

    # Print the measurement
    print(f"Grain {i}: {length_px:.2f} pixels")

print("Total grains measured:", len(rice_contours))

cv2.imshow("Measured Rice Grains", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
