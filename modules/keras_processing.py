from keras.layers import Input, Activation, Dense, Dropout, BatchNormalization, Conv3D, MaxPooling3D, Flatten
from keras.layers import concatenate, add
from keras.optimizers import adam

from keras.models import Model

from keras import backend as K


def swish(x):
    return (K.sigmoid(x) * x)


class ConvPeriodic3D(Conv3D):
    """
        Custom 3D convolution layer with periodic boundary conditions
         - on each side, append extremity points with the same value as the opposite side
         - then simply apply normal convolution with "valid" condition
    """
    def build(self, input_shape):
        super(ConvPeriodic3D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
                self.filters)

    def call(self, x):
        if self.kernel_size[0] == 2:
            x = K.concatenate([x, x[:, :1, :, :, :]], axis=1)
        if self.kernel_size[0] == 3:
            x = K.concatenate([x[:, -1:, :, :, :], x, x[:, :1, :, :, :]], axis=1)
        if self.kernel_size[1] == 2:
            x = K.concatenate([x, x[:, :, :1, :, :]], axis=2)
        if self.kernel_size[1] == 3:
            x = K.concatenate([x[:, :, -1:, :, :], x, x[:, :, :1, :, :]], axis=2)
        if self.kernel_size[2] == 2:
            x = K.concatenate([x, x[:, :, :, :1, :]], axis=3)
        if self.kernel_size[2] == 3:
            x = K.concatenate([x[:, :, :, -1:, :], x, x[:, :, :, :1, :]], axis=3)

        return super(ConvPeriodic3D, self).call(x)


def make_model(dimX, nb, nlayer, nunit, pdropout):
    """
        Make model comprised of:
            - CNN with custom Conv3D layers with periodic conditions and skip connections to process cube
            - Dense + Dropout layers with swish activations to process output of CNN  + X
    """
    input_cube = Input(shape=(nb, nb, nb, 4))

    output_conv = input_cube
    for ilayer, nunit_layer in enumerate([5, 10, 15, 20, 30, 40, 50, 75, 100]):
        output_2x2 = ConvPeriodic3D(nunit_layer, (2, 1, 1), padding='valid')(output_conv)
        output_2x2 = ConvPeriodic3D(nunit_layer, (1, 2, 1), padding='valid')(output_2x2)
        output_2x2 = ConvPeriodic3D(nunit_layer, (1, 1, 2), padding='valid')(output_2x2)
        output_2x2 = BatchNormalization()(output_2x2)

        output_1x1 = Conv3D(nunit_layer, (1, 1, 1), padding='same')(output_conv)

        output_conv = add([output_1x1, output_2x2])
        output_conv = Activation(swish)(output_conv)

        if ilayer % 3 == 2:
            output_2x2 = ConvPeriodic3D(nunit_layer, (2, 1, 1), padding='valid')(output_conv)
            output_2x2 = ConvPeriodic3D(nunit_layer, (1, 2, 1), padding='valid')(output_2x2)
            output_2x2 = ConvPeriodic3D(nunit_layer, (1, 1, 2), padding='valid')(output_2x2)
            output_2x2 = MaxPooling3D(pool_size=(2, 2, 2), padding='valid')(output_2x2)
            output_2x2 = BatchNormalization()(output_2x2)

            output_1x1 = Conv3D(nunit_layer, (1, 1, 1), padding='same')(output_conv)
            output_1x1 = MaxPooling3D(pool_size=(2, 2, 2), padding='valid')(output_1x1)
            output_1x1 = BatchNormalization()(output_1x1)

            output_conv = add([output_1x1, output_2x2])
            output_conv = Activation(swish)(output_conv)

    output_conv = Flatten()(output_conv)

    # merge with X
    input_X = Input(shape=(dimX,))
    output = concatenate([input_X, output_conv])

    for idense in range(nlayer):
        output = Dense(nunit)(output)
        output = Activation(swish)(output)
        output = Dropout(pdropout)(output)

    output = Dense(2)(output)
    model = Model(inputs=[input_cube, input_X], outputs=[output])

    # 'mse' to optimize the same metric as the competition
    model.compile(loss='mse', optimizer=adam(clipnorm=0.001, lr=0.001,  epsilon=1e-08, decay=0.0))
    return model
