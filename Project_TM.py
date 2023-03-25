import cv2
import numpy as np

# Load the target image and the template image
target_img = cv2.imread('Main_image.jpg')
template_img = cv2.imread('f16_template.jpg')

# cv2.namedWindow('InitialImage', cv2.WINDOW_NORMAL)
cv2.imshow('InitialImage', target_img)
cv2.waitKey(0)
cv2.destroyWindow('InitialImage')

# Convert the images to grayscale
target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('GreyImage', cv2.WINDOW_NORMAL)
cv2.imshow('GreyImage', target_gray)
cv2.waitKey(0)
cv2.destroyWindow('GreyImage')


# Apply Gaussian blur to both the images
target_blur = cv2.GaussianBlur(target_gray, (5, 5), 0)
template_blur = cv2.GaussianBlur(template_gray, (5, 5), 0)

# cv2.namedWindow('BlurImage', cv2.WINDOW_NORMAL)
cv2.imshow('BlurImage', target_blur)
cv2.waitKey(0)
cv2.destroyWindow('BlurImage')

# Apply edge detection algorithm to the grayscale,blurred target image
target_edges = cv2.Canny(target_blur, 50, 200)
template_edges = cv2.Canny(template_blur, 50, 200)

# cv2.namedWindow('EdgeImage', cv2.WINDOW_NORMAL)
cv2.imshow('EdgeImage', target_edges)
cv2.waitKey(0)
cv2.destroyWindow('EdgeImage')

# Apply template matching algorithm to the edge-detected target image
# result = cv2.matchTemplate(target_edges, template_edges, cv2.TM_SQDIFF_NORMED)

# # print the result matrix
# print(result.shape)
# for i in range(result.shape[0]):
#     for j in range(result.shape[1]):
#         if(result[i][j] > 0.5):
#             print(result[i][j],"position: ", i, j)

# perform template matching using the normalized squared difference method without using the library
result = np.zeros((target_edges.shape[0] - template_edges.shape[0] + 1, target_edges.shape[1] - template_edges.shape[1] + 1))
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        result[i][j] = np.sum(np.square(target_edges[i:i+template_blur.shape[0], j:j+template_blur.shape[1]] - template_edges))

print(np.argmin(result))
min_loc = np.unravel_index(np.argmin(result), result.shape)
# print(min_loc)
min_loc = (min_loc[1], min_loc[0])
print(min_loc)

# Locate the highest matching score
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# print(min_loc)

# Draw a rectangle around the area of the target image that matches the template
left_loc = min_loc
right_loc = (left_loc[0] + template_img.shape[1], left_loc[1] + template_img.shape[0])
cv2.rectangle(target_img, left_loc, right_loc, (0, 0, 255), 2)

# Display the original target image with the rectangle drawn around the matching area
cv2.imshow('Target Image with Matching Area', target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




