import cv2
import numpy as np

def load_image(path):
    img_rgb  = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return img_rgb, img_gray

def apply_sampling(img_gray):
    h, w = img_gray.shape
    scales = [0.25, 0.5, 1.0, 1.5, 2.0]
    results = []
    for s in scales:
        ds  = cv2.resize(img_gray, (int(w*s), int(h*s)))
        out = cv2.resize(ds, (w, h))
        results.append(out)
    return results

def apply_quantization(img_gray):
    results = []
    for bits in [8, 4, 2]:
        levels = 2 ** bits
        q = np.round(img_gray / 255 * (levels-1)) * (255/(levels-1))
        results.append(q.astype(np.uint8))
    return results

def apply_intensity_transforms(img_gray):
    I = img_gray.astype(np.float64)
    neg  = (255 - I).astype(np.uint8)
    c    = 255 / np.log(1 + 255)
    log  = (c * np.log(1 + I)).astype(np.uint8)
    g05  = (255 * (I/255) ** 0.5).astype(np.uint8)
    g15  = (255 * (I/255) ** 1.5).astype(np.uint8)
    return neg, log, g05, g15

def process_image(img_gray):
    step1    = cv2.GaussianBlur(img_gray, (5,5), 1)
    step2    = (255 * (step1.astype(np.float64)/255)**0.5).astype(np.uint8)
    step3    = cv2.equalizeHist(step2)
    kernel   = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    enhanced = cv2.filter2D(step3, -1, kernel)
    return enhanced
