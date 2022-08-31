
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms


# Convert (C H W) RGB image to (C H w) YCbCr image
def rgb2ycbcr(im):

    print(im)
    r: torch.Tensor = im[0,:,:]
    g: torch.Tensor = im[1,:,:]
    b: torch.Tensor = im[2,:,:]

    print('rgb2ycbcr')
    print(r.shape)
    print(r)
    print(g.shape)
    print(g)
    print(b.shape)
    print(b)

    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = 128 - r * 0.1687 - g * 0.3313 + b * 0.5
    cr: torch.Tensor = 128 + r * 0.5 - g * 0.4187 - b * 0.0813 

    print(y.shape)
    print(y)
    print(cb.shape)
    print(cb)
    print(cr.shape)
    print(cr)
    
    return y, cb, cr

def ycbcr2rgb(im):

    y: torch.Tensor = im[0,:,:]
    cb: torch.Tensor = im[1,:,:]
    cr: torch.Tensor = im[2,:,:]

    r = y + 1.402 * cr - 128 * 1.402
    g = y -  0.3414 * cb + 128 * 0.3414 - 0.71414 * cr + 128 * 0.7141
    b = y +  1.772 * cb - 128 * 1.772 

    return r, g, b


class SuperResolutionPreProcess(nn.Module):

    def __init__(self):
        super(SuperResolutionPreProcess, self).__init__()
        self.resize = transforms.Resize([224, 224])

    def forward(self, img):

        # Image passed in base64 format from application
        arr = np.frombuffer(base64.b64decode(img), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        print(img.shape)

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
            
        print(f'Image shape after load {img.shape}')
        print(f'Text pixel {RGB_img[3,2]}')
    
        # Swap channel and pixel axes to input that resize expects
        img = img.permute(2,0,1)

        print(f'C H W image {img}')

        # Resize image to 224x224
        img = self.resize(img)
        print(f'Image shape after resize {img.shape}')

        display_img = img.permute(1,2,0)
        plt.imshow(display_img)
        plt.show()

        y, cb, cr = rgb2ycbcr(img)

        ycbcr = torch.stack((y, cb, cr))
        ycbcr = ycbcr.permute(1,2,0).type(torch.uint8)
        print(f' Clamped {ycbcr}')
        print(f'ycbcr shape {ycbcr.shape}')
        plt.imshow(ycbcr)
        plt.show()

        return y, cb, cr

        
class SuperResolutionPostProcess(nn.Module):

    def __init__(self):
        super(SuperResolutionPostProcess, self).__init__()

    def forward(self, y, cb, cr):

        y = y.squeeze()

        print(y.shape)

        # Resize the cb and cr dimensions
        c = torch.stack((cb, cr)).unsqueeze(0)
        c = F.interpolate(c, y.size()[0]).squeeze()

        print(c.shape)

        # Add the y axes back into the image
        img = torch.cat((c, y.unsqueeze(0)))

        return ycbcr2rgb(img)

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.preprocess = SuperResolutionPreProcess()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.postprocess = SuperResolutionPostProcess()

        self._initialize_weights()

    def forward(self, img):
        y, cb, cr = self.preprocess(img)
        y = y.unsqueeze(0)

        print(f'y shape after pre process: {y.shape}')

        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.pixel_shuffle(self.conv4(y))

        print(f'y shape before post process {y.shape}')

        r, g, b = self.postprocess(y, cb, cr)
        return torch.stack((r, g, b))

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)

# Load pretrained model weights
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()

# load the image
image = cv2.imread('cat_224x224.jpg')

print(f'Image shape after cv2.imread {image.shape}')
RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(RGB_img.shape)

print(f'Text pixel {RGB_img[3,2]}')
plt.imshow(RGB_img)
plt.show()

retval, buffer_img= cv2.imencode('.png', RGB_img)
data = base64.b64encode(buffer_img)

img = torch_model(data)

print(img.shape)

plt.imshow(img.detach().permute(1, 2, 0)  )
plt.show()





