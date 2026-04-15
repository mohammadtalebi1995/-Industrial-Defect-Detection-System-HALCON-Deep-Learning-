import cv2

def preprocess(img):
    img = cv2.GaussianBlur(img, (5,5), 0)

    # simulate illumination correction
    background = cv2.blur(img, (25,25))
    corrected = cv2.subtract(img, background)

    return corrected