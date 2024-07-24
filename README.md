import cv2
import numpy as np

# Load degraded image
degraded_img = cv2.imread('degraded_image.jpg', cv2.IMREAD_GRAYSCALE)

# Optionally, preprocess (e.g., denoise if needed)
# degraded_img = cv2.medianBlur(degraded_img, 5)

# Define the PSF (or estimate it)
psf = np.ones((5, 5)) / 25.0  # Example PSF for motion blur

# Apply Wiener filter for restoration
restored_img = cv2.filter2D(degraded_img, -1, psf)
restored_img = cv2.warpAffine(restored_img, psf, (degraded_img.shape[1], degraded_img.shape[0]))

# Optionally, postprocess (e.g., enhance contrast)
# restored_img = cv2.equalizeHist(restored_img)

# Display or save the restored image
cv2.imshow('Restored Image', restored_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
