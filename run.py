import argparse

import cv2
import imutils
import numpy as np
from keras.preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator

from config import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, VALIDATION_DATA_DIR, BATCH_SIZE
from model import ClassificationNet

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

validation_datagen = ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

classes = validation_generator.class_indices

# load the trained convolutional neural network
print("[INFO] loading network...")
model = ClassificationNet.build((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), 2)

model.load_weights(args["model"])

# classify the input image
res = model.predict(image)[0]

label = None
percentage = 40.0

for cls, index in classes.items():
    formatting_perc =  res[index]*100
    print('{}: {:4.2f}%'.format(cls, formatting_perc))
    if formatting_perc >= percentage:
        percentage = formatting_perc
        label = cls

label = 'unknown' if not label else label

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
