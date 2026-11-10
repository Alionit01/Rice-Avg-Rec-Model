
import cv2
import numpy as np

# 1. Load image
image = cv2.imread("data/rice.jpeg")

if image is None:
    raise FileNotFoundError("Could not load data/rice.jpg")

# 2. Resize image
image = cv2.resize(image, (1020, 768))

# 3. Calibration
pixels_per_mm = 4.845

# 4. Define Region of Interest
ROI_Y = 350

# Only analyse the area below the ruler
roi = image[ROI_Y:768, 0:1020]

# 5. Convert ROI to grayscale
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# 6. Create binary mask
_, mask = cv2.threshold(
    gray, 100, 255, cv2.THRESH_BINARY
)

# 7. Remove noise
kernel = np.ones((3, 3), np.uint8)

clean_mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN, kernel
)

# 8. Find contours
contours, _ = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 9. Filter contours
min_area = 100

rice_contours = [
    contour for contour in contours
    if cv2.contourArea(contour) >= min_area
]

# 10. Measure grains
grain_lengths_mm = []
result = image.copy()

for i, contour in enumerate(rice_contours, start=1):

    rect = cv2.minAreaRect(contour)

    width, height = rect[1]

    length_px = max(width, height)

    length_mm = length_px / pixels_per_mm

    grain_lengths_mm.append(length_mm)

    # Draw rotated rectangle
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # Shift coordinates back to the original image
    box[:, 1] += ROI_Y

    cv2.drawContours(
        result, [box], 0, (0, 255, 0), 2
    )

    print(f"Grain {i}: {length_mm:.2f} mm")

# 11. Calculate average
if grain_lengths_mm:

    average_length = np.mean(grain_lengths_mm)

    print("\n--- RiceCheck Results ---")
    print("Total grains:", len(grain_lengths_mm))
    print("Average grain length:", round(average_length, 2), "mm")

    cv2.putText(
        result,
        f"Average: {average_length:.2f} mm",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

else:
    print("No grains detected.")

# 12. Display results
cv2.imshow("RiceCheck Results", result)
cv2.imshow("ROI Mask", clean_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
