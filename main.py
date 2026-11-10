
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

    # Shift coordinates back to original image
    box[:, 1] += ROI_Y

    # Draw green bounding box
    cv2.drawContours(
        result, [box], 0, (0, 255, 0), 2
    )

    # Get label position using shifted coordinates
    x = int(np.min(box[:, 0]))
    y = int(np.min(box[:, 1]))

    label = f"{length_mm:.1f} mm"

    cv2.putText(
        result,
        label,
        (x, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA
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

# 12. Rice Quality est
if average_length < 5.5:
    rice_type = "Short Grain"
elif average_length <= 6.5:
    rice_type = "Medium Grain"
else:
    rice_type = "Long Grain"

print(f"Rice Category: {rice_type}")

shortest_grain = min(grain_lengths_mm)
longest_grain = max(grain_lengths_mm)


short_count = sum(1 for x in grain_lengths_mm if x < 6)
medium_count = sum(1 for x in grain_lengths_mm if 6 <= x <= 7.5)
long_count = sum(1 for x in grain_lengths_mm if x > 7.5)

# 13. Add results panel

panel_height = 190
panel = np.zeros((panel_height, 1020, 3), dtype=np.uint8)

cv2.putText(panel, "RICECHECK RESULTS",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2)

cv2.putText(panel, f"Total grains: {len(grain_lengths_mm)}",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1)

cv2.putText(panel, f"Average length: {average_length:.2f} mm",
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1)

cv2.putText(panel, f"Shortest: {shortest_grain:.2f} mm",
            (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1)

cv2.putText(panel, f"Longest: {longest_grain:.2f} mm",
            (20, 175), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1)

cv2.putText(panel, f"Category: {rice_type}",
            (550, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)

cv2.putText(panel, f"Short: {short_count}  Medium: {medium_count}",
            (550, 115), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1)

cv2.putText(panel, f"Long: {long_count}",
            (550, 150), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1)

# Combine panel with original result image
final_result = np.vstack((result, panel))

cv2.imshow("RiceCheck - Final Results", final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 14. Display results
cv2.imshow("ROI Mask", clean_mask)

print(f"Shortest grain: {shortest_grain:.2f} mm")
print(f"Longest grain: {longest_grain:.2f} mm")


cv2.imshow("Rice Grain Measurements", result)

print("\n--- Grain Distribution ---")
print(f"Short grains: {short_count}")
print(f"Medium grains: {medium_count}")
print(f"Long grains: {long_count}")

cv2.waitKey(0)
cv2.destroyAllWindows()
