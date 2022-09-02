
from ctypes.wintypes import RGB
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Convert (C H W) RGB image to (C H w) YCbCr image
def rgb2ycbcr(im):

    r: torch.Tensor = im[0,:,:]
    g: torch.Tensor = im[1,:,:]
    b: torch.Tensor = im[2,:,:]

    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = 0.5 - r * 0.1687 - g * 0.3313 + b * 0.5
    cr: torch.Tensor = 0.5 + r * 0.5 - g * 0.4187 - b * 0.0813 

    return y, cb, cr

def ycbcr2rgb(im):

    y: torch.Tensor = im[0,:,:]
    cb: torch.Tensor = im[1,:,:]
    cr: torch.Tensor = im[2,:,:]

    r = y + 1.402 * cr - 0.5 * 1.402
    g = y -  0.3414 * cb + 0.5 * 0.3414 - 0.71414 * cr + 0.5 * 0.7141
    b = y +  1.772 * cb - 0.5 * 1.772 

    return r, g, b

def test_ycbcr2rgb():

    # input = torch.tensor([[[255]],[[255]],[[255]]])
    # y, cb, cr = rgb2ycbcr(input)
    # print('ycbcr of white')
    # print(y)
    # print(cb)
    # print(cr)

    # input = torch.tensor([[[0]],[[0]],[[0]]])
    # y, cb, cr = rgb2ycbcr(input)
    # print('y, cb, cr of black')
    # print(y)
    # print(cb)
    # print(cr)

    # input = torch.tensor([[[255]],[[0]],[[0]]])
    # y, cb, cr = rgb2ycbcr(input)
    # print('y, cb, cr of red')
    # print(y)
    # print(cb)
    # print(cr)

    # input = torch.tensor([[[0]],[[255]],[[0]]])
    # y, cb, cr = rgb2ycbcr(input)
    # print('y, cb, cr of green')
    # print(y)
    # print(cb)
    # print(cr)

    # input = torch.tensor([[[0]],[[0]],[[255]]])
    # y, cb, cr = rgb2ycbcr(input)
    # print('y, cb, cr of blue')
    # print(y)
    # print(cb)
    # print(cr)

    # Random
    input = torch.tensor([[[70]],[[134]],[[210]]])
    y, cb, cr = rgb2ycbcr(input)
    print('y, cb, cr of random')
    print(y)
    print(cb)
    print(cr)

    r, g, b = ycbcr2rgb(torch.stack((y, cb, cr)))
    print('RGB of random')
    print(r)
    print(g)
    print(b)

def img_from_y(y, size):
    c = torch.full((2, size[1], size[2]), 0.5)
    return torch.cat((y, c))

def show_img(img, title):
    print(f'Displaying image: {title}')    
    display_img = img.permute(1,2,0)
    plt.imshow(display_img.detach().numpy())
    plt.title(title)
    plt.show()

class SuperResolutionPreProcess(nn.Module):

    def __init__(self):
        super(SuperResolutionPreProcess, self).__init__()

    # Accepts a base64 encoded image.
    # Returns the 3 channels of the image in YCbCr encoding
    def forward(self, img):

        # Image passed in base64 format from application
        arr = np.frombuffer(base64.b64decode(img), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)

        # Swap channel and pixel axes C H W
        img = img.permute(2,0,1)

        show_img(img, 'After normalize, before YCbCr')

        y, cb, cr = rgb2ycbcr(img)

        ycbcr = torch.stack((y, cb, cr))
        show_img(ycbcr, 'YCbCr after preprocess')

        return y, cb, cr

        
class SuperResolutionPostProcess(nn.Module):

    def __init__(self):
        super(SuperResolutionPostProcess, self).__init__()


    # Accepts the YCbCr channels of the expanded resolution image
    # Returns the RGB channels
    def forward(self, y, cb, cr):

        y = y.squeeze()

        # Resize the cb and cr dimensions
        c = torch.stack((cb, cr)).unsqueeze(0)

        print(c.shape)

        # Resize image to 224*3 x 224*3
        self.resize = transforms.Resize([c.shape[2]*3, c.shape[3]*3])
        c = self.resize(c).squeeze()

        print(c.shape)

        # Add the y axes back into the image
        img = torch.cat((y.unsqueeze(0), c))

        # Show image at end of preprocess
        show_img(img, 'YCbCr after postprocess')

        return ycbcr2rgb(img)

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):

        print(x.shape)
        img = img_from_y(x, x.shape)
        show_img(img, 'Luminance only before super resolution')

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))

        img = img_from_y(x, x.shape)
        show_img(img, 'Luminance only after super resolution')

        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

class SuperResolutionPipeline(torch.nn.Module):

    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionPipeline, self).__init__()
        self.preprocess = SuperResolutionPreProcess()
        self.net = SuperResolutionNet(upscale_factor=3)
        self.postprocess = SuperResolutionPostProcess()

    def forward(self, img):
        y, cb, cr = self.preprocess(img)
        y = self.net(y.unsqueeze(0))
        r, g, b = self.postprocess(y, cb, cr)
        return torch.stack((r, g, b))


    def net_model(self):
        return self.net



# Test utility functions first
#test_ycbcr2rgb()

# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionPipeline(upscale_factor=3)

# Load pretrained model weights
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.net_model().load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()

# load the image
image = cv2.imread('cat_224x224.jpg')
RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(RGB_img)
plt.title('Original image')
plt.show()

retval, buffer_img= cv2.imencode('.jpg', RGB_img)
data = base64.b64encode(buffer_img)

img = torch_model(data)

show_img(img, 'Final image')

# Export the model to ONNX
torch.onnx.export()






