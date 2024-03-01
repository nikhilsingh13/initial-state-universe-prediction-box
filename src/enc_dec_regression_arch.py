from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D

def enc_dec_regression(input_shape):
    # Building the model
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input_shape)
    x = MaxPooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((2, 2, 2), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((2, 2, 2))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((2, 2, 2))(x)
    decoded = Conv3D(1, (3, 3, 3), activation='linear', padding='same')(x)  # linear activation function for regression

    ae_model = Model(input_shape, decoded)