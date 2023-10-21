import numpy as np
import cv2

# Sample arrays (replace with yours)
array1 = np.array([[10, 10], [30, 50], [50, 90]])
array2 = np.array([[90, 10], [70, 50], [30, 90]])

# Create a blank image
h, w = 100, 100
image = np.zeros((h, w, 3), dtype=np.uint8)

# Color of the line for drawing original lines
color = (0, 255, 0)  # Green
thickness = 2

# Draw lines
for (x1, y1), (x2, y2) in zip(array1, array2):
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# Convert image to grayscale for Hough transform
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Hough transform
lines = cv2.HoughLines(gray_image, 1, np.pi / 180, 50)

# Draw the detected lines
for rho, theta in lines[:, 0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Draw detected lines in blue

# Display the image
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
