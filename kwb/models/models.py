from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Permute, Reshape,Lambda,Conv2D, MaxPooling2D,GlobalAveragePooling2D,UpSampling2D,BatchNormalization
from keras.regularizers import l1_l2,l1,l2
from keras.constraints import non_neg


def iris_mlp(weights = "../../examples/resources/models/iris_mlp.hdf5"):
    model = Sequential()
    model.add(Dense(8, input_shape=(4,),name="firstlayer"))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    model.add(Dense(8,name="secondlayer"))
    model.add(Dropout(0.1))
    model.add(Activation("relu"))
    model.add(Dense(3,name="prepredictions"))
    model.add(Activation("softmax",name="predictions"))
    model.compile("adam",loss="categorical_crossentropy",metrics=["acc"])
    model.load_weights(weights)
    return model

def dna(weights = "../../examples/resources/models/dna_cnn.hdf5"):
    model = Sequential()
    model.add(Conv2D(64,(1,3),
                     padding="same",
                     kernel_regularizer=l1_l2(),
                     input_shape=(1,2000,4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(128,(1,3),
                     padding="same",
                     kernel_regularizer=l1_l2()))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(128,(1,3),
                     padding="same",
                     kernel_regularizer=l1_l2()))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(256,(1,3),
                     padding="same",
                     kernel_regularizer=l1_l2()))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(512,(1,3),
                     padding="same",
                     kernel_regularizer=l1_l2()))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,2)))

    model.add(Conv2D(512,(1,3),
                     padding="same",
                     kernel_regularizer=l1_l2()))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,2)))


    model.add(GlobalAveragePooling2D())

    model.add(Dense(512,kernel_regularizer=l2()))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Dense(1,kernel_regularizer=l2(),name="prepredictions"))
    model.add(Activation("sigmoid",name="predictions"))
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',metrics=["acc"])
    model.load_weights(weights)
    return model

