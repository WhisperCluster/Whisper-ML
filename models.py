from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization, GRU
from keras.optimizers import SGD

def compile_fp_model_active():
    """
    Builds and compiles a GRU-based neural network for frequency classification.

    This model consists of three stacked GRU layers with 1024 units each,
    ReLU activations, and dropout regularization. All GRU layers return
    sequences, followed by a dense softmax output layer for categorical
    prediction over 480 frequency bin.

    The model is compiled using the Adam optimizer and categorical
    cross-entropy loss, with accuracy as the evaluation metric.

    Returns
    -------
    tensorflow.keras.models.Sequential
        A compiled Keras Sequential model ready for training.
    """

    model = Sequential()

    # GRU Layer 1
    model.add(
        GRU(
            1024,
            return_sequences=True,
            activation='relu',
            input_shape=(1, 480)
        )
    )
    model.add(Dropout(0.3))

    # GRU Layer 2
    model.add(
        GRU(
            1024,
            return_sequences=True,
            activation='relu'
        )
    )
    model.add(Dropout(0.3))

    # GRU Layer 3
    model.add(
        GRU(
            1024,
            return_sequences=True,  
            activation='relu'
        )
    )
    model.add(Dropout(0.3))

    # Dense Output
    model.add(Dense(480, activation='softmax'))

    # Compile the model with optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def compile_sw_ms_model_active_natural():
    """
    Builds and compiles a fully connected neural network for multi-region classification

    This model is a deep  neural network composed of four dense layers
    with batch normalization and dropout for regularization. ReLU activations are
    used in the hidden layers, and a sigmoid activation is applied in the output
    layer to support multi-label predictions across three classes.

    Returns
    -------
    tensorflow.keras.models.Sequential
        A compiled Keras Sequential model ready for training.
    """

    model = Sequential()

    # Layer 1
    model.add(Dense(2048, 
                    input_shape=(950,), 
                    kernel_initializer='uniform', 
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 2
    model.add(Dense(1024, 
                    kernel_initializer='normal', 
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 3
    model.add(Dense(512, 
                    kernel_initializer='normal', 
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 4
    model.add(Dense(3, 
                    kernel_initializer='uniform', 
                    activation='sigmoid'))

    # Optimizer: Stochastic Gradient Descent (SGD)
    sgd = SGD(learning_rate=0.0005, momentum=0.9, decay=1e-6, nesterov=True)
    
    # Compile the model with optimizer
    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
        )
    
    return model 

def compile_ps_model_natural():
    """
    Builds and compiles a fully connected neural network for binary region classification.

    This model consists of two hidden dense layers with ReLU activations,
    batch normalization, and dropout regularization, followed by a sigmoid
    output layer with two units for binary or multi-label prediction.

    Returns
    -------
    tensorflow.keras.models.Sequential
        A compiled Keras Sequential model ready for training.
    """

    model = Sequential()
    # Layer 1 
    model.add(Dense(
        1024,
        input_shape=(470,),
        kernel_initializer='uniform',
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 2
    model.add(Dense(
        512,
        kernel_initializer='uniform',
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 3 (output)
    model.add(Dense(
        2,
        kernel_initializer='uniform',
        activation='sigmoid'
    ))

    # Optimizer: Stochastic Gradient Descent (SGD)
    sgd = SGD(
        learning_rate=0.001,
        momentum=0.9,
        nesterov=True,
    decay=1e-6,
    )

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model 

def compile_tl_model_natural():
    """
    Builds and compiles a fully connected neural network for binary region classification.

    This model consists of two hidden dense layers with ReLU activations,
    batch normalization, and dropout regularization, followed by a sigmoid
    output layer with two units for binary or multi-label prediction.

    Returns
    -------
    tensorflow.keras.models.Sequential
        A compiled Keras Sequential model ready for training.
    """
    model = Sequential()

    # Layer 1
    model.add(Dense(
        512,
        input_shape=(470,),
        kernel_initializer='uniform',
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 2
    model.add(Dense(
        480,
        kernel_initializer='uniform',  
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 3 (output)
    model.add(Dense(
        2,
        kernel_initializer='uniform',
        activation='sigmoid'
    ))

    # Optimizer: Stochastic Gradient Descent (SGD)
    sgd = SGD(
        learning_rate=0.001,
        momentum=0.9,
        nesterov=True,
        decay=1e-6
    )

    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    
    return model 
