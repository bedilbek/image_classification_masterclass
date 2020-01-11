from keras import Input, Model, Sequential
from keras import backend as K
from keras.layers import Conv2D, Dropout, Flatten, Dense


class TrafficSignNet:

    @staticmethod
    def build(width_height_channel):

        (width, height, channel) = width_height_channel
        if K.image_data_format() == 'channels_first':
            input_shape = (channel, height, width)
        else:
            input_shape = (height, width, channel)

        model = Sequential(name='TrafficSignNet')
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Dropout(0.15))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.15))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.15))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, activation='sigmoid', name='traffic_sign_output'))

        return model
