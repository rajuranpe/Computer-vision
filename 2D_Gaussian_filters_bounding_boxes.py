import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_and_draw_bounding_boxes_sobel_upgrade(image, blur_kernel_size, sigma_x, sigma_y, min_area):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur with different sigma values to reduce noise and emphasize horizontal edges
    gaussian_kernel_x = cv2.getGaussianKernel(ksize=blur_kernel_size, sigma=sigma_x)
    gaussian_kernel_y = cv2.getGaussianKernel(ksize=blur_kernel_size, sigma=sigma_y)
    gaussian_kernel = gaussian_kernel_y @ gaussian_kernel_x.T  # create 2D kernel
    blurred_image = cv2.filter2D(gray_image, -1, gaussian_kernel)

    # apply sobel operator in the y direction (horizontal edges)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)

    # convert sobel output to 8-bit unsigned integer and threshold
    sobel_y_8u = np.uint8(np.abs(sobel_y))
    _, binary_image = cv2.threshold(sobel_y_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours on binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw bounding boxes around the contours on the original image
    image_with_boxes = image.copy()
    pixel_counts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            pixel_counts.append(area)

    return image_with_boxes, pixel_counts

# Parameters
blur_kernel_size = 21  # Kernel size, adjust if needed
sigma_x = 5  # Horizontal spread
sigma_y = 2  # Vertical spread, should be less than sigma_x for horizontal emphasis
min_area = 3000  # Minimum area threshold to consider for a object of interest (also max_area, as a portion of the whole image resolution?)

# Load the image
image_path = r'path\to\input_file'
image = cv2.imread(image_path)

result_image, pixel_counts = find_and_draw_bounding_boxes_sobel_upgrade(image, blur_kernel_size, sigma_x, sigma_y, min_area)

# display the result
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title(f"Blur: {blur_kernel_size}, Min Area: {min_area}")
plt.axis('off')
plt.show()

# print the number of pixels within each bounding box
print("Pixels within each bounding box:", pixel_counts)

image_path = r'C:path\to\input_file'
image = cv2.imread(image_path)

# Check
if image is None:
    print("Error loading image")

# apply horizontal emphasis filter
def horizontal_emphasis_filter(image, blur_kernel_size, sigma_x, sigma_y):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur with a large sigma in x and small in y
    gaussian_kernel_x = cv2.getGaussianKernel(ksize=blur_kernel_size, sigma=sigma_x)
    gaussian_kernel_y = cv2.getGaussianKernel(ksize=blur_kernel_size, sigma=sigma_y)
    gaussian_kernel = gaussian_kernel_y @ gaussian_kernel_x.T  # Create 2D kernel
    blurred_image = cv2.filter2D(gray_image, -1, gaussian_kernel)
    
    return blurred_image

# Generate a mask from the emphasized image
def generate_mask_from_emphasis(emphasized_image, threshold=100):
    # Apply a binary threshold to image
    _, binary_mask = cv2.threshold(emphasized_image, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_mask

# Apply horizontal filter
emphasized_image = horizontal_emphasis_filter(image, blur_kernel_size=10, sigma_x=20, sigma_y=2)

# Generate a mask from the image
binary_mask = generate_mask_from_emphasis(emphasized_image)

# Visualize
plt.figure(figsize=(18, 6))

#plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

def draw_refined_bounding_boxes(image, mask):
    # invert the mask
    inverted_mask = cv2.bitwise_not(mask)
    
    # Find contours
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_boxes = image.copy()

    for contour in contours:
        # Calculate the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out the bounding boxes for noise, like text or dirt. Assuming the objects will have a certain minimum width and height
        if w > 50 and h > 50:  # might need changes
            # box in red
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image_with_boxes

# Load the original image
original_image = cv2.imread(r'path\to\input_file')

# Draw bounding boxes
image_with_refined_boxes = draw_refined_bounding_boxes(original_image, binary_mask)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image_with_refined_boxes, cv2.COLOR_BGR2RGB))
plt.title('Original Image with BB')
plt.axis('off')
plt.show()
