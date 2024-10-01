import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# Load the image
image = cv2.imread(r'C:\Users\Uporabnik\source\repos\KeyPixelImageCompression\Datasets\COCO\000000000001.jpg')
# image = cv2.imread(r'C:\Users\Uporabnik\source\repos\KeyPixelImageCompression\Datasets\DIV2K\Original\0774.png')

# plt.figure()
# plt.subplot(231)
# plt.imshow(image, cmap="gray")


start = time.time()

for i in range(10):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Tresholding
    thresh = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1)
    # _, thresh = cv2.threshold(gray_blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # plt.subplot(232)
    # plt.imshow(thresh, cmap="gray")

    # Noise removal using Morphological Transformations
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)

    # plt.subplot(233)
    # plt.imshow(opening, cmap="gray")

    # Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # plt.subplot(234)
    # plt.imshow(dist_transform, cmap="gray")

    # Treshold distance transform
    _, sure_fg = cv2.threshold(dist_transform, 5, 255, cv2.THRESH_BINARY)
    # plt.subplot(235)
    # plt.imshow(sure_fg, cmap="gray")

    # Marker labelling
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    # plt.subplot(236)
    # plt.imshow(markers, cmap="gray")

    # Apply watershed
    markers = np.int32(markers)
    markers_w = cv2.watershed(image, markers)

end = time.time()
print((end - start) / 10.0)



# Create an empty image to display segments in different colors
segments = np.zeros_like(image)

# Assign random colors to each segment based on markers
np.random.seed(3)
segment_sizes = np.zeros(np.unique(markers)[-1] + 1)
cntSegments = 0
for marker in np.unique(markers):
    if marker == -1:
        continue  # Skip the watershed boundary
    mask = markers == marker
    segment_sizes[marker] = np.count_nonzero(mask)
    segments[mask] = np.random.randint(0, 255, 3)
    cntSegments += 1

for y in np.arange(segments.shape[0]):
    for x in np.arange(segments.shape[1]):
        if markers[y][x] == -1:
            segments[y][x] = np.random.randint(0, 255, 3)

segment_sizes = segment_sizes[segment_sizes != 0]
segment_sizes = np.append(segment_sizes, np.ones(len(markers[markers == -1])))

print(f'Avg. segment size: {np.average(segment_sizes)} +- {np.std(segment_sizes)}')

# # Display the segmented regions
plt.figure()
plt.subplot(121)
plt.imshow(markers_w)
plt.title('Segments with Random Colors: ' + str(len(segment_sizes)))
plt.axis('off')

kernel = np.ones((3, 3), np.uint8)
# segments = cv2.dilate(segments, kernel, iterations=1)
plt.subplot(122)
plt.imshow(segments)
plt.show()

# plt.imsave('out.png', segments)
