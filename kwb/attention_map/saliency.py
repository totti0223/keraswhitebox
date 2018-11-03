import numpy as np
import keras.backend as K
from . misc import deprocess_image

def saliency_map(input_model, images, layer_name="predictions",cls=-1,normalize="img_standard"):
    '''
    function to draw vanilla backpropagation or guided backpropagation
    depends on the input_model
    '''

    #batch_tolerance
    if len(images.shape) == 3 or len(images.shape) == 1:
        images = np.expand_dims(images,axis=0)

    grads_val = []

    for i in range(images.shape[0]):
        if cls == -1:
            _cls = np.argmax(input_model.predict(images[i:i+1]))
        else:
            _cls = cls

        input_imgs = input_model.input
        #layer_output =  input_model.get_layer(layer_name).output
        layer_output = input_model.get_layer(layer_name).output[0,_cls]
        #max_output = K.max(layer_output, axis=1)
        #grads = K.gradients(K.sum(max_output), input_imgs)[0]
        grads = K.gradients(layer_output,input_imgs)[0]
        backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])

        _grads_val = backprop_fn([images[i:i+1], 0])[0][0]
        
        if normalize == "raw":
            pass
        elif normalize == "img_standard":
            _grads_val = deprocess_image(_grads_val)
        elif normalize == "abs":
            _grads_val = np.abs(_grads_val).max(axis=-1) / _grads_val.max()
        elif normalize == "pos":
            _grads_val = np.maximum(0, _grads_val) / _grads_val.max()
        elif normalize == "neg":
            _grads_val *= -1
            _grads_val = np.maximum(0, _grads_val) / _grads_val.max()
        grads_val.append(_grads_val)
    grads_val = np.array(grads_val)

    #if not batch, return without batch dimension
    if grads_val.shape[0] == 1:
        grads_val = grads_val[0]

    #delete backprop_fn to free memory
    del backprop_fn
    
    return grads_val