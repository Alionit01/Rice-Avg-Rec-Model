
import cv2
import numpy as np

# 1. Load image
image = cv2.imread("data/rice.jpeg")

if image is None:
    raise FileNotFoundError("Could not load data/rice.jpg")

# 2. Resize image
image = cv2.resize(image, (1020, 768))

# 3. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. Create binary mask
_, mask = cv2.threshold(
    gray, 100, 255, cv2.THRESH_BINARY
)

# 5. Remove noise
kernel = np.ones((3, 3), np.uint8)
clean_mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN, kernel
)

# 6. Find contours
contours, _ = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 7. Filter contours
min_area = 100

rice_contours = [
    contour for contour in contours
    if cv2.contourArea(contour) >= min_area
]

# 8. Measure grain lengths
grain_lengths = []
result = image.copy()

for i, contour in enumerate(rice_contours, start=1):

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]

    length_px = max(width, height)
    grain_lengths.append(length_px)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.drawContours(
        result, [box], 0, (0, 255, 0), 2
    )

# 9. Calculate average
if grain_lengths:
    average_length = np.mean(grain_lengths)

    print("Total grains:", len(grain_lengths))
    print("Average grain length:", round(average_length, 2), "pixels")

    cv2.putText(
        result,
        f"Average: {average_length:.2f} pixels",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )
else:
    print("No grains detected.")

# 10. Display result
cv2.imshow("Rice Grain Measurements", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

#grid_image = result.copy()

# Draw vertical lines every 10 pixels
#for x in range(0, 1020, 10):
#   cv2.line(grid_image, (x, 0), (x, 768), (255, 255, 255), 1)

# Draw horizontal lines every 10 pixels
#for y in range(0, 768, 10):
#    cv2.line(grid_image, (0, y), (1020, y), (255, 255, 255), 1)

#cv2.imshow("Rice Image - Pixel Grid", grid_image)