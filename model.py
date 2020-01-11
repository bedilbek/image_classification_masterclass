from keras import Input, Model
from keras import backend as K
from keras.layers import Conv2D, Dropout, Flatten, Dense


class TrafficSignNet:

    @staticmethod
    def build(width_height_channel, num_classes):

        (width, height, channel) = width_height_channel
        if K.image_data_format() == 'channels_first':
            input_shape = (channel, height, width)
        else:
            input_shape = (height, width, channel)

        inputs = Input(shape=input_shape)

        x = Conv2D(32, kernel_size=(5, 5), strides=2, activation='relu')(inputs)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(num_classes, activation='softmax', name='traffic_sign_output')(x)

        model = Model(inputs=inputs, outputs=x, name='TrafficSignNet')

        return model
