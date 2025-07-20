from pathlib import Path
import numpy as np
import cv2 as cv
import pytesseract
from imutils.perspective import four_point_transform

# Optional: Set this if Tesseract is not in your system PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
image_path = r"D:\python files\doc_scanner project\images\03.jpg"
img_org = cv.imread(image_path)

if img_org is None:
    print(f"[ERROR] Could not load image from path: {image_path}")
    exit()

# Resize image to a manageable size for processing
def resize(img, width=500):
    h, w, _ = img.shape
    height = int((h / w) * width)
    size = (width, height)
    resized_img = cv.resize(img, size)
    return resized_img, size

# Resize and enhance
re_img, size = resize(img_org)
img_en = cv.detailEnhance(re_img, sigma_r=0.15, sigma_s=20)

# Convert to grayscale, blur, and detect edges
gray = cv.cvtColor(img_en, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
edge = cv.Canny(blur, 75, 200)

# Morphological transformations to close gaps
kernel = np.ones((5, 5), np.uint8)
dilate = cv.dilate(edge, kernel, iterations=1)
closing = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)

# Find contours and sort by area
contours, _ = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)

# Try to find a 4-point contour
four_points = None
for contour in contours:
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        four_points = np.squeeze(approx)
        break

# If no 4-point contour found, exit
if four_points is None:
    print("[ERROR] Document contour not detected.")
    exit()

# Scale the points to match original image size
multiplier = img_org.shape[1] / size[0]
four_points_orgi = (four_points * multiplier).astype(int)

# Warp original image using perspective transform
wrap = four_point_transform(img_org, four_points_orgi)

# OCR: Extract text using Tesseract
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(wrap, config=custom_config)

# Save extracted text to file
output_file = Path("scanned_text.txt")
with output_file.open("w", encoding='utf-8') as f:
    f.write(text)

# Print preview
print("[INFO] Text extraction complete. Check the 'scanned_text.txt' file.\n")
print("Preview of extracted text:\n")
print(text[:500] + ("\n..." if len(text) > 500 else ""))

# Show the final scanned document
cv.namedWindow("Scanned Document", cv.WINDOW_NORMAL)
cv.imshow("Scanned Document", wrap)
cv.waitKey(0)
cv.destroyAllWindows()
