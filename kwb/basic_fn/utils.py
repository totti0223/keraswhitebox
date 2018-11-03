from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import six

import time
import json
import warnings

try:
    import requests
except ImportError:
    requests = None

import itertools
import os
import tempfile
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

import keras
import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras import activations
from keras.models import Model,load_model

from skimage.io import imread, imshow
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom

def show_inference(model,image,label="imagenet",topn=5):
    '''
    visualize inference as matplot graph.

    Args:
        model (keras model)
        image : shape (height,width,channel)
        label : only compatible for imagenet now.
        topn: top inference to show
    Returns:
        None. Will just draw a graph via plt.show()

    '''
    _image = np.expand_dims(image,axis=0)
    result = model.predict(_image)
    
    if label=="imagenet":
        decoded_result = np.array(decode_predictions(result,top=topn)[0])
        idents = [x for x in decoded_result[:,0]] 
        names = [x for x in decoded_result[:,1]] 
        vals = [float(x)*100 for x in decoded_result[:,2]]
    plt.title("Result")
    plt.barh(range(len(vals)),vals[::-1],alpha=0.2)

    plt.tick_params(
        axis='y',
        labelleft=False)
    if vals[::-1][0]<100:
        plt.xlabel("Probability (%)")
        plt.xlim([0,100])
        for i, (name, val) in enumerate(zip(names[::-1],vals[::-1])):
            plt.text(s=" ".join([name,'{:.1f}'.format(val)+"%"]), x=2, y=i, color="black", verticalalignment="center", size=14)
    else:
        plt.xlabel("A.U.")
        for i, (name, val) in enumerate(zip(names[::-1],vals[::-1])):
            plt.text(s=" ".join([name,str(int(val))]), x=2, y=i, color="black", verticalalignment="center", size=14)
    plt.show()

def show_imgs(imgs,cmap="jet",row=6, col=6,sortby=None,layer_name=""):
    """Show numpy images as row*col
    #modified based on https://qiita.com/takurooo/items/c06365dd43914c253240
    Arguments:
            imgs: numpy array, (height,width,channel) or (height,width,channel,batch), as is output of keras 
            row: Int, row for plt.subplot
            col: Int, column for plt.subplot
    Returns:
            None.
    """
    if len(imgs.shape)==4:
        imgs = imgs.transpose(3,0,1,2)
    if len(imgs.shape)==3:
        imgs = imgs.transpose(2,0,1)
    
    indices = range(len(imgs))

    if len(imgs) > (row * col):
        print("number of image exceeds row*col. first %s images will be drawn" % (row*col))
    plt.figure(figsize=(8, 8)) 
    #plt.suptitle(layer_name)
    for i, img in zip(indices,imgs):
        if i == (row*col):
            break
        plot_num = i+1
        plt.subplot(row, col, plot_num)
        plt.title("Filter:%s" % i,size=12)
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(deprocess_image(img),cmap=cmap)
    plt.tight_layout()

def show_1d_heatmap(vals,cmap="jet",layer_name=""):
    '''
    Show numpy images as 1d heatmap
    Arguments:
            vals: 1-D numpy array,
    Returns:
            None.
    '''
    a = np.expand_dims(vals,axis=0)
    fig, ax = plt.subplots(figsize=(8,1))
    #plt.suptitle(layer_name)
    heatmap = ax.pcolor(a, cmap=cmap)
    ax.set_xticks(np.arange(a.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(np.arange(a.shape[1]))
    ax.set_yticks([])
    plt.show()

def first_conv_filters(model,row=6,col=6,visualize=True):
    '''
    Will scan through the model and get values of first conv filters. 
    examples: weights = first_conv_filters(model)
                weights = first_conv_filters(model, visualize=False)
    Args:
        model (keras model)
        image : shape (height,width,channel)
        label : only compatible for imagenet now.
        topn: top inference to show
        visualize: Will output filter images via plt.show()
    Returns:
        numpy array of weights. (height,weight,channels,number of filters) 

    '''
    #obtain the first conv2d layer.
    layer_dict = OrderedDict([(layer.name, layer) for layer in model.layers])
    for name,ltype in layer_dict.items():
        if "Conv" in str(ltype):
            layer_name = name
            print("Attempting to visualize the first conv2d layer : %s" % layer_name)
            break

    weights = layer_dict[layer_name].get_weights()[0]
    if visualize:
        show_imgs(weights)
    return weights

def intermediate_output(model,images,layer_name,visualize=True,row=6,col=6,cmap="jet",sortby=None):
    '''
    Will create a intermediate_output of the fed image with the model and designated layer_name 
    Args:
        model (keras model)
        images: shape (height,width,channel) or (batch,height,width,channel)
            OR for MLP, (vals) or (batch, vals)
        layar_name: refer from model.summary()
        visualize: Will output filter images via plt.show()
        row,col: default 6. when visualized, the no. images per col and row.
        sortby: not ready to use yet.
    Returns:
        numpy array of intermediate outputs. 
            if 2d output...(height,weight,channels,number of filters)
            if 1d output...(values, number of filters)
        Note: if input image has batch dimention, output will also harbor batch dimention

    '''
    #batch tolerance
    if len(images.shape) == 3:
        images = np.expand_dims(images,axis=0)
    if len(images.shape) == 1:
        images = np.expand_dims(images,axis=0)

    layer_dict = OrderedDict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[layer_name].output
    
    input_img = model.input
    intmodel = Model(inputs=input_img,outputs=layer_output)
    intout = intmodel.predict(images)
    
    for i in range(intout.shape[0]):
        if visualize == True:
            if len(intout[i].shape) == 3:
                show_imgs(intout[i],cmap=cmap,row=row,col=col,layer_name=layer_name,sortby=sortby)
            else:
                show_1d_heatmap(intout[i],cmap=cmap,layer_name=layer_name)
    
    #if not batch, return without batch dimension
    if images.shape[0] == 1:
        intout = intout[0]

    return intout

def weights_histogram(model,layer_name):
    layer_dict = OrderedDict([(layer.name, layer) for layer in model.layers])
    weights = layer_dict[layer_name].get_weights()[0]
    plt.hist(weights.flatten())
    plt.title("weights histogram of %s" % layer_name)

def deprocess_image(x):
        x = x.copy()
        if np.ndim(x) > 3:
            x = np.squeeze(x)
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_dim_ordering() == 'th':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    #from the sklearn examples
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.    
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class Callback(object):
    """Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


# ==========================================================================
#
#  Multi-GPU Model Save Callback
#
# ==========================================================================

class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)