from PIL import Image
import numpy as np

image = Image.open("data/test/0/test_3.png")
# image = Image.open("data/129.png")
array = np.array(image)
image2 = Image.fromarray(array)