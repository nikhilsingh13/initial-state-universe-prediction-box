from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Add, Activation, Dropout

def conv_block(input_tensor, num_filters):
    shortcut = Conv3D(num_filters, (1,1,1), padding='same')(input_tensor) # skip connection added for residual connection
    x = Conv3D(num_filters, (3,3,3), padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Conv3D(num_filters, (3,3,3), padding='same')(x)
    x = Activation('relu')(x)
    return Add()([x, shortcut]) # residual connection as per the VNET paper64

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling3D((2,2,2))(x)
    p = Dropout(0.5)(p)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    x = UpSampling3D((2,2,2))(input_tensor)
    x = Conv3D(num_filters, (1,1,1))(x) # 1X1X1 convolution operation to match the filters
    x = Add()([x, concat_tensor]) # skip connection implementation. VNET is a variant of UNET.
    x = conv_block(x, num_filters)
    return x

def vnet_regression(input_shape):
    inputs = Input(input_shape)
    
    e0, p0 = encoder_block(inputs, 16)
    e1, p1 = encoder_block(p0, 32)
    e2, p2 = encoder_block(p1, 64)
    e3, p3 = encoder_block(p2, 128)
    
    b0 = conv_block(p3, 256)
    
    d3 = decoder_block(b0, e3, 128)
    d2 = decoder_block(d3, e2, 64)
    d1 = decoder_block(d2, e1, 32)
    d0 = decoder_block(d1, e0, 16)
    
    outputs = Conv3D(1, (1,1,1), activation='linear')(d0)
    
    model = Model(inputs=[inputs], outputs=[outputs], name="vnet_regression")
    
    return model