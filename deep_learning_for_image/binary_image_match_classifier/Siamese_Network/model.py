from keras import backend as K
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
)
from keras.models import Model, Sequential
from keras.regularizers import l2
import numpy as np


# def create_cnn(width, height, depth):
# 	inputShape = (height, width, depth)
# 	inputs = Input(shape=inputShape)

# 	x = Conv2D(32, (3, 3), activation="relu", padding="same",
# 				kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4))(inputs)
# 	x = BatchNormalization()(x)

# 	x = Conv2D(32, (3, 3), activation="relu", padding="same",
# 				kernel_initializer=initialize_weights,bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(x)
# 	x = BatchNormalization()(x)
# 	x = MaxPooling2D(pool_size=(2, 2))(x)
# 	# x = Activation("relu")(x)
# 	x = Dropout(0.3)(x)

# 	x = Conv2D(64, (3, 3), activation="relu", padding="same",
# 				kernel_initializer=initialize_weights,bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(x)
# 	x = BatchNormalization()(x)
# 	x = Conv2D(64, (3, 3), activation="relu", padding="same",
# 				kernel_initializer=initialize_weights,bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(x)
# 	x = BatchNormalization()(x)
# 	x = MaxPooling2D(pool_size=(2, 2))(x)
# 	# x = Activation("relu")(x)
# 	x = Dropout(0.3)(x)

# 	x = Conv2D(128, (3, 3), activation="relu", padding="same",
# 				kernel_initializer=initialize_weights,bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(x)
# 	x = BatchNormalization()(x)
# 	x = MaxPooling2D(pool_size=(2, 2))(x)
# 	# x = Activation("relu")(x)
# 	x = Dropout(0.2)(x)


# 	x = Conv2D(4096, (3, 3), activation="relu", padding="same",
# 				kernel_initializer=initialize_weights,bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(x)
# 	x = BatchNormalization()(x)
# 	# x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
# 	# x = BatchNormalization()(x)
# 	x = MaxPooling2D(pool_size=(2, 2))(x)
# 	# x = Activation("relu")(x)
# 	x = Dropout(0.5)(x)

# 	x = Flatten()(x)
# 	x = Dense(4096, activation="relu",
# 				kernel_regularizer=l2(1e-3),kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(x)

# 	encoded_l = Model(inputs, x)
# 	encoded_r = Model(inputs, x)

# 	L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
# 	L1_distance = L1_layer([encoded_l, encoded_r])
# 	prediction = Dense(1,activation="sigmoid",bias_initializer=initialize_bias)(L1_distance)

# 	siamese_model = Model(inputs=[encoded_l.input, encoded_l.input], outputs=prediction)

# 	return siamese_model


def create_cnn(depth: int, height: int, width: int) -> Model:
    """
    Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    input_shape = (width, height, depth)

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=input_shape,
            kernel_initializer=initialize_weights,
            kernel_regularizer=l2(2e-4),
        )
    )
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(BatchNormalization())
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer=initialize_weights,
            bias_initializer=initialize_bias,
            kernel_regularizer=l2(2e-4),
        )
    )
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer=initialize_weights,
            bias_initializer=initialize_bias,
            kernel_regularizer=l2(2e-4),
        )
    )
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer=initialize_weights,
            bias_initializer=initialize_bias,
            kernel_regularizer=l2(2e-4),
        )
    )

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(
        Dense(
            4096,
            activation="sigmoid",
            kernel_regularizer=l2(1e-3),
            kernel_initializer=initialize_weights,
            bias_initializer=initialize_bias,
        )
    )

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation="sigmoid", bias_initializer=initialize_bias)(
        L1_distance
    )

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


def initialize_bias(shape) -> np.ndarray:
    """
    The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def initialize_weights(shape) -> np.ndarray:
    """
    The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)
