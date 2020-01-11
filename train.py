import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator, np

from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL, TRAIN_DATA_DIR, BATCH_SIZE, VALIDATION_DATA_DIR, \
    NB_TRAIN_SAMPLES, EPOCHS, NB_VALIDATION_SAMPLES
from model import ClassificationNet

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

model = ClassificationNet.build((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), validation_generator.num_classes)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

H = model.fit_generator(
    train_generator,
    steps_per_epoch=NB_TRAIN_SAMPLES // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=NB_VALIDATION_SAMPLES // BATCH_SIZE
)

model.save_weights('model_saved.h5')

# plot the total loss, throttle loss, steering_right loss and steering_left loss
print("[INFO] visualizing losses and accuracies over epochs...")
lossNames = ['loss']
plt.style.use("ggplot")
(fig, ax) = plt.subplots()

# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax.set_title(title)
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss")
    ax.plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax.plot(np.arange(0, EPOCHS), H.history["val_" + l],
            label="val_" + l)
    ax.legend()

# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plt.savefig("classification_model_losses.png")
plt.close()

# create a new figure for the accuracies
accuracyNames = ['accuracy']
plt.style.use("ggplot")
(fig, ax) = plt.subplots()

# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
    ax.set_title("Accuracy for {}".format(l))
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Accuracy")
    ax.plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax.plot(np.arange(0, EPOCHS), H.history["val_" + l],
            label="val_" + l)
    ax.legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("classification_model_accs.png")
plt.close()
