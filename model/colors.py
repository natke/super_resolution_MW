import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import base64

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T))

def show(im):
    plt.imshow(im)
    plt.show()

im = Image.open('test_pattern.jpg').convert('YCbCr')
im = np.array(im)
print(im.shape, im.dtype)

im = cv2.imread('test_pattern.jpg')
print(im.shape, im.dtype)
print(im.max(), im.min())
show(im)

im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
print(im_ycrcb.shape, im_ycrcb.dtype)
print(im_ycrcb.max(), im_ycrcb.min())
show(im_ycrcb)

im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCrCb2BGR)
print(im_rgb.shape, im_rgb.dtype)
print(im_rgb.max(), im_rgb.min())
show(im_rgb)

image = cv2.imread('test_pattern.png')
print(image)
retval, buffer_img= cv2.imencode('.png', image)
data = base64.b64encode(buffer_img)
print(data)
nparr = np.frombuffer(base64.b64decode(data), np.uint8)
decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

im_ycrcb = rgb2ycbcr(decoded_image)
show(im_ycrcb)

im_rgb = ycbcr2rgb(im_ycrcb)
show(im_rgb)
