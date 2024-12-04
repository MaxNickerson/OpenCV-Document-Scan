import streamlit as st
import cv2
import numpy as np
from PIL import Image


# function to resize image
def resize_image(image, max_height=2000):
    height, width = image.shape[:2]
    if height > max_height:
        ratio = max_height / float(height)
        new_dimensions = (int(width * ratio), max_height)
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        return image, ratio
    return image, 1.0

def remove_text(image, kernel_size=(5, 5), iterations=3):
    # grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create a kernal for mophological operations
    kernel = np.ones(kernel_size, np.uint8)

    # apply closing operation to remove small objects (text)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # convert back to BGR for compatibility with further processing
    image_no_text = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

    return image_no_text

def remove_background(image, iterations=5):
    # mask initialization
    mask = np.zeros(image.shape[:2], np.uint8)

    # models used by grabcut algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # define the rectangle that contains the document
    height, width = image.shape[:2]
    rect = (20,20, width - 40, height - 40)

    # apply grabcut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)

    # Create a mask where background pixels are set to 0, foreground to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # apply the mask to the image
    image_no_bg = image * mask2[:, :, np.newaxis]

    return image_no_bg

def detect_edges(image, low_threshold=50, high_threshold=150, aperture_size=3):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)

    # dilate the edges to strengthen them (optional)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def find_document_contour(edges):
    # finds the largest contour in the edge-detected image that approximates to a quadrilateral.

    # find contours in the edge-detected image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # initialize the document contour
    document_contour = None

     # loop over the contours to find the one that approximates to a quadrilateral
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02* peri, True)

        # If the approximated contour has four points, we can assume it's the document
        if len(approx) == 4:
            document_contour = approx
            break

    return document_contour

def get_ordered_corners(contour):
    # Reshape the contour to a 4x2 array
    contour = contour.reshape(4, 2)

    # Initialize a list to store the ordered corners
    ordered_corners = np.zeros((4, 2), dtype='float32')

    # Calculate the sum of the coordinates and diff
    s = contour.sum(axis=1)
    diff = np.diff(contour, axis=1)

   # The top-left point will have the smallest sum
    ordered_corners[0] = contour[np.argmin(s)]
    # The bottom-right point will have the largest sum
    ordered_corners[2] = contour[np.argmax(s)]
    # The top-right point will have the smallest difference
    ordered_corners[1] = contour[np.argmin(diff)]
    # The bottom-left point will have the largest difference
    ordered_corners[3] = contour[np.argmax(diff)]

    return ordered_corners

def get_transformation_matrix(corners):
    (tl, tr, br, bl) = corners

    # compute width of the new image, which will be the maximum distance between bottoms and btwn tops
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    # compute height of new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # define the destination points which will be used to map the document to a top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

     # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(corners, dst)

    return M, maxWidth, maxHeight

def apply_perspective_transform(orig_image, M, maxWidth, maxHeight):
    warped = cv2.warpPerspective(orig_image, M, (maxWidth, maxHeight))

    return warped

def main():
    st.title('Document Scanner - Preprocessing Test')

    # Sidebar controls for morphological operations
    st.sidebar.header('Morphological Operations Parameters')
    kernel_size = st.sidebar.slider('Kernel Size', min_value=1, max_value=15, value=5, step=2)
    iterations_morph = st.sidebar.slider('Iterations', min_value=1, max_value=10, value=3)

    # Sidebar controls for GrabCut
    st.sidebar.header('GrabCut Parameters')
    iterations_grabcut = st.sidebar.slider('GrabCut Iterations', min_value=1, max_value=10, value=5)

    # Sidebar controls for Edge Detection
    st.sidebar.header('Canny Edge Detection Parameters')
    low_threshold = st.sidebar.slider('Low Threshold', min_value=0, max_value=100, value=50)
    high_threshold = st.sidebar.slider('High Threshold', min_value=100, max_value=300, value=150)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image of a document", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        orig_image = image.copy()

        # Display the original image
        st.subheader('Original Image')
        st.image(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB), channels='RGB')

        # Step 1: Resize the image
        image, resize_ratio = resize_image(image)
        st.subheader('Resized Image')
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels='RGB')

        # Step 2: Remove text with morphological operations
        image_no_text = remove_text(image, kernel_size=(kernel_size, kernel_size), iterations=iterations_morph)
        st.subheader('After Removing Text with Morphological Operations')
        st.image(cv2.cvtColor(image_no_text, cv2.COLOR_BGR2RGB), channels='RGB')

        # Step 3: Remove background with GrabCut
        image_no_bg = remove_background(image_no_text, iterations=iterations_grabcut)
        st.subheader('After Background Removal with GrabCut')
        st.image(cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2RGB), channels='RGB')

        # Step 4: Detect edges
        edges = detect_edges(image_no_bg, low_threshold=low_threshold, high_threshold=high_threshold)
        st.subheader('Edge Detection')
        st.image(edges, channels='GRAY')

        # Step 5: Find document contour
        document_contour = find_document_contour(edges)

        # Check if a document contour was found
        if document_contour is not None:
            # Rescale the contour coordinates to match the original image size
            document_contour_scaled = document_contour.astype('float') / resize_ratio
            document_contour_scaled = document_contour_scaled.astype('int')

            # Draw the contour on the original image
            contour_image = orig_image.copy()
            cv2.drawContours(contour_image, [document_contour_scaled], -1, (0, 255, 0), 2)
            st.subheader('Document Contour')
            st.image(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB), channels='RGB')

            # Step 6: Get ordered corners
            ordered_corners = get_ordered_corners(document_contour_scaled)

            # Step 7: Get transformation matrix
            M, maxWidth, maxHeight = get_transformation_matrix(ordered_corners)

            # Step 8: Apply perspective transformation
            scanned = apply_perspective_transform(orig_image, M, maxWidth, maxHeight)
            st.subheader('Scanned Document')
            st.image(cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB), channels='RGB')

            # Optionally, save the scanned document
            # cv2.imwrite('scanned_document.jpg', scanned)
        else:
            st.write("Could not find document contour. Please adjust the parameters or try a different image.")




if __name__ == "__main__":
    main()