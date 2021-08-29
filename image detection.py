import numpy as np
import cv2
from pprint import pprint as pp
from operator import itemgetter
import sys

BLOCK_SIZE = 4

# Load image
image = cv2.imread(str(sys.argv[1]), cv2.IMREAD_COLOR)
image = cv2.imread(r"C:\Users\ACER\Desktop\project image manipulation\forged1.png")
cv2.imshow('', image)

cv2.waitKey(1)
height, width = image.shape[:2]

# Create matrix block_pixels
block_pixels = []
index = 0
for row in range(0, height - BLOCK_SIZE + 1):
    for col in range(0, width - BLOCK_SIZE + 1):
        roi = image[row : row + BLOCK_SIZE, col : col + BLOCK_SIZE]
        block_row = [list(j) for sub in roi for j in sub]
        block_pixels.append([index, block_row])
        index += 1

# Sort matrix
sorted_block_pixels = sorted(block_pixels, key=itemgetter(1))

# Find matching blocks
matched_indexes = []
for i in range(len(sorted_block_pixels)-1):
    if(sorted_block_pixels[i][1] == sorted_block_pixels[i+1][1]):
        matched_indexes.append([sorted_block_pixels[i][0], sorted_block_pixels[i+1][0]])
        print(matched_indexes[-1])

# Mark forged parts of the image
result = np.zeros((height, width, 3), np.uint8)
forged = np.zeros((BLOCK_SIZE, BLOCK_SIZE, 3), np.uint8)
forged[:,:] = (255,255,255)

for i in matched_indexes:
    h0 = int(i[0]/(width - BLOCK_SIZE + 1))
    w0 = int(i[0]%(width - BLOCK_SIZE + 1))
    result[h0 : h0 + BLOCK_SIZE, w0 : w0 + BLOCK_SIZE] = forged
    h1 = int(i[1]/(width - BLOCK_SIZE + 1))
    w1 = int(i[1]%(width - BLOCK_SIZE + 1))
    result[h1 : h1 + BLOCK_SIZE, w1 : w1 + BLOCK_SIZE] = forged

cv2.imshow("Forged Parts", result)

cv2.waitKey(0)
cv2.destroyAllWindows()