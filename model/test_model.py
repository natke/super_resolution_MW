import cv2
import numpy as np
import onnxruntime
import base64

# load the image
image = cv2.imread('test_pattern.png')
print(image)

# test to and from base64
retval, buffer_img= cv2.imencode('.png', image)
data = base64.b64encode(buffer_img)
print(data)

# Decode the base64 image
nparr = np.frombuffer(base64.b64decode(data), np.uint8)
decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
print(decoded_image)
cv2.imwrite("test_patterm_out.png", decoded_image)


image_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(image)

# Load the model
#session = onnxruntime.InferenceSession('mobilenetv2-12-aug.onnx')

# Run the model
#results = session.run(["top_classes", "top_probs"], {"image": image_ortvalue})

#print(results)