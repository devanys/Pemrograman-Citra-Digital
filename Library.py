import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage, fft, signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import exposure, filters, feature, morphology, segmentation, measure
from skimage.color import rgb2hsv, rgb2lab, rgb2gray, label2rgb
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, auc, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
