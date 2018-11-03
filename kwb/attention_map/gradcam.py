import numpy as np
from scipy.ndimage.interpolation import zoom

import keras.backend as K

def grad_cam(input_model, images, layer_name, cls=-1, method = "naive",resize_to_input=True):

    #check input shape first. whether its a batch or not.
    batch = True
    if len(images.shape) == 3:
        images = np.expand_dims(images,axis=0)
        batch = False
    
    #image shape will be (batch,H,W,channel)
    H = images.shape[1]
    W = images.shape[2]

    cam = []

    for i in range(images.shape[0]):
        if cls == -1:
            _cls = np.argmax(input_model.predict(images[i:i+1]))
        else:
            _cls = cls

        y_c = input_model.output[0, _cls]
        conv_output = input_model.get_layer(layer_name).output

        #print(i)
        if method == "naive":
            grads = K.gradients(y_c, conv_output)[0]
            gradient_function = K.function([input_model.input], [conv_output, grads])
            _output, _grads_val = gradient_function([images[i:i+1]])
            _output, _grads_val = _output[0,:,:,:], _grads_val[0, :, :, :]
            _weights = np.mean(_grads_val, axis=(0, 1))
            _cam = np.dot(_output, _weights)
        elif method == "gradcampp":
            grads = K.gradients(y_c, conv_output)[0]
            first = K.exp(y_c)*grads
            second = K.exp(y_c)*grads*grads
            third = K.exp(y_c)*grads*grads
            gradient_function = K.function([input_model.input], [first,second,third, conv_output, grads])

            conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([images[i:i+1]])
            global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)
            alpha_num = conv_second_grad[0]
            alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
            alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
            alphas = alpha_num/alpha_denom
            _weights = np.maximum(conv_first_grad[0], 0.0)
            alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)
            alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))
            deep_linearization_weights = np.sum((_weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
            _cam = np.sum(deep_linearization_weights*conv_output[0], axis=2)
        #scale 0 to 1 rather than clip 0 then divide by max
        #_cam = (_cam-_cam.min())/(_cam.max()-_cam.min())
        _cam = np.maximum(_cam,0)
        #_cam = _cam / _cam.max()
        if resize_to_input:
            #_cam = resize(_cam, (H, W))
            _cam = zoom(_cam,H/_cam.shape[0])
        cam.append(_cam)
    cam = np.array(cam)
    
    #if not batch, return without batch dimension
    if batch == False:
        cam = cam[0]
    del gradient_function
    return cam