import cv2
import numpy as np

from image_handlers.edge_detect import CannyEdgeDetector, Laplacian


def thresh_otsu(image_np):
    blur = cv2.GaussianBlur(image_np, (5, 5), 0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return otsu


def contour_mask2(image_np, area_filter=0.9):
    h, w = image_np.shape[:2]
    max_area = h * w * area_filter
    shape = image_np.shape
    if len(shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY_INV, 11, 2)

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: cv2.contourArea(c))
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contours[-1]], 255)
    return mask


def contour_mask(image_np, area_filter=0.9):
    h, w = image_np.shape[:2]
    max_area = h * w * area_filter
    shape = image_np.shape
    if len(shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                            cv2.THRESH_BINARY_INV, 11, 2)
    thresh = thresh_otsu(gray)

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: cv2.contourArea(c))
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contours[-1]], 255)
    return mask


canny = CannyEdgeDetector()
laplace = Laplacian()
sub1 = cv2.createBackgroundSubtractorKNN(history=1000)
sub2 = cv2.createBackgroundSubtractorMOG2(history=1000)

source = cv2.VideoCapture("shoe.mp4")
ret, frame = source.read()
kernel = np.ones((3, 3), np.uint8)

while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask1 = sub1.apply(frame)
    _, mask1 = cv2.threshold(mask1, 10, 255, cv2.THRESH_BINARY)


    mask2 = sub2.apply(frame)
    _, mask2 = cv2.threshold(mask2, 10, 255, cv2.THRESH_BINARY)

    mask3 = contour_mask(frame)
    mask4 = contour_mask2(frame)
    mask5 = thresh_otsu(gray)

    mask6 = canny.apply(frame)
    mask6 = cv2.cvtColor(mask6, cv2.COLOR_BGR2GRAY)

    mask7 = cv2.Laplacian(gray, cv2.CV_64F)
    _, mask7 = cv2.threshold(mask7, 8, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(mask7.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: cv2.contourArea(c))
    mask7 = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask7, [contours[-1]], 255)

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    final = np.average([mask1, mask2, mask3, mask4, mask5, mask6, mask7, sx, sy], axis=0)

    frame[final < 65] = (0, 0, 0)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    ret, frame = source.read()
source.release()

cv2.destroyAllWindows()
