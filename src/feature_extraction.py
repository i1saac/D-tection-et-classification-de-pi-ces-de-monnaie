import cv2
import numpy as np
from skimage.filters import gabor
from skimage.feature import hog

def extract_color_features(img_hsv):
         hist_h = cv2.calcHist([img_hsv], [0], None, [64], [0, 180])
         hist_s = cv2.calcHist([img_hsv], [1], None, [64], [0, 256])
         hist_v = cv2.calcHist([img_hsv], [2], None, [64], [0, 256])
         hist_h = cv2.normalize(hist_h, hist_h).flatten()
         hist_s = cv2.normalize(hist_s, hist_s).flatten()
         hist_v = cv2.normalize(hist_v, hist_v).flatten()
         return np.concatenate([hist_h, hist_s, hist_v])

def extract_texture_features(img_gray):
         frequencies = [0.1, 0.3, 0.5]
         thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
         features = []
         for freq in frequencies:
             for theta in thetas:
                 filt_real, _ = gabor(img_gray, frequency=freq, theta=theta)
                 features.append(filt_real.mean())
                 features.append(filt_real.std())
         return np.array(features)

def extract_contour_features(img_gray):
         edges = cv2.Canny(img_gray, 100, 200)
         return np.array([np.mean(edges), np.std(edges)])

def extract_shape_features(img_gray):
         moments = cv2.HuMoments(cv2.moments(img_gray)).flatten()
         return moments

def extract_features(img_hsv, img_gray):
         color_features = extract_color_features(img_hsv)
         texture_features = extract_texture_features(img_gray)
         contour_features = extract_contour_features(img_gray)
         shape_features = extract_shape_features(img_gray)
         return np.concatenate([color_features, texture_features, contour_features, shape_features])