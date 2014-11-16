import numpy as np
from PIL import Image

img = Image.open("2000px_navy_blue.png").convert('RGB')
a_img = np.asarray(img, dtype="uint8")
print a_img
print a_img.shape
