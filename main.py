# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
from PIL import Image
import io

# Function to resize image
def resize_image(image, max_height=2000):
    height, width = image.shape[:2]
    if height > max_height:
        ratio = max_height / float(height)
        new_dimensions = (int(width * ratio), max_height)
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        return image, ratio
    return image, 1.0

def remove_text(image, kernel_size=(5, 5), iterations=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    image_no_text = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    return image_no_text

def remove_background(image, iterations=5):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = image.shape[:2]
    rect = (20,20, width - 40, height - 40)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image_no_bg = image * mask2[:, :, np.newaxis]
    return image_no_bg

def detect_edges(image, low_threshold=50, high_threshold=150, aperture_size=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges

def find_document_contour(edges):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    document_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02* peri, True)
        if len(approx) == 4:
            document_contour = approx
            break
    return document_contour

def get_ordered_corners(contour):
    contour = contour.reshape(4, 2)
    ordered_corners = np.zeros((4, 2), dtype='float32')
    s = contour.sum(axis=1)
    diff = np.diff(contour, axis=1)
    ordered_corners[0] = contour[np.argmin(s)]
    ordered_corners[2] = contour[np.argmax(s)]
    ordered_corners[1] = contour[np.argmin(diff)]
    ordered_corners[3] = contour[np.argmax(diff)]
    return ordered_corners

def get_transformation_matrix(corners):
    (tl, tr, br, bl) = corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return M, maxWidth, maxHeight

def apply_perspective_transform(orig_image, M, maxWidth, maxHeight):
    warped = cv2.warpPerspective(orig_image, M, (maxWidth, maxHeight))
    return warped

# Initialize the FastAPI app
app = FastAPI()

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = Image.open(file.file)
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        orig_image = image.copy()

        # Parameters
        kernel_size = (5, 5)
        iterations_morph = 3
        iterations_grabcut = 5
        low_threshold = 50
        high_threshold = 150

        # Step 1: Resize the image
        image, resize_ratio = resize_image(image)

        # Step 2: Remove text with morphological operations
        image_no_text = remove_text(
            image, kernel_size=kernel_size, iterations=iterations_morph)

        # Step 3: Remove background with GrabCut
        image_no_bg = remove_background(
            image_no_text, iterations=iterations_grabcut)

        # Step 4: Detect edges
        edges = detect_edges(
            image_no_bg, low_threshold=low_threshold, high_threshold=high_threshold)

        # Step 5: Find document contour
        document_contour = find_document_contour(edges)

        # Check if a document contour was found
        if document_contour is not None:
            # Rescale the contour coordinates to match the original image size
            document_contour_scaled = document_contour.astype(
                'float') / resize_ratio
            document_contour_scaled = document_contour_scaled.astype('int')

            # Step 6: Get ordered corners
            ordered_corners = get_ordered_corners(document_contour_scaled)

            # Step 7: Get transformation matrix
            M, maxWidth, maxHeight = get_transformation_matrix(ordered_corners)

            # Step 8: Apply perspective transformation
            scanned = apply_perspective_transform(
                orig_image, M, maxWidth, maxHeight)

            # Convert the image to bytes
            scanned_rgb = cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(scanned_rgb)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)

            return StreamingResponse(img_bytes, media_type="image/jpeg")
        else:
            return JSONResponse(content={"error": "Could not find document contour. Please adjust the parameters or try a different image."}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
