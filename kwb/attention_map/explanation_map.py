import numpy as np
from scipy.ndimage.interpolation import zoom
from keras.models import Model
def EM(model,target_image,reference_images,layername="activation_1"):    
    '''
    reference based attention map generating algorithms proposed in
    An explainable deep machine vision framework for plant stress phenotyping
    Sambuddha Ghosala,1, David Blystoneb,1, Asheesh K. Singhb, Baskar Ganapathysubramaniana, Arti Singhb,2, and Soumik Sarkara,2
    input:
        model : keras model
        target_image : colored input image with or without batch dimention
        regerence_images : reference_image(s) with or without needes batch dimention
        layername : layer of interest.
    returns: heatmap
    '''
    if len(target_image.shape) == 3:
        target_image = np.expand_dims(target_image,axis=0)
    if len(reference_images.shape) == 3:
        reference_images = np.expand_dims(reference_images,axis=0)
    
    #get intermediate outputs of reference data    
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layername).output)
    intermediate_output = intermediate_layer_model.predict(reference_images)
    
    #print(intermediate_output.shape) ----> batch,height,width,channel  #only for convolutional output with height width dimentions.
    '''
    threshold = []
    for img in intermediate_output:
        for i in range(intermediate_output.shape[3]):
            feature = img[:,:,i]
            means = np.mean(feature)         
            threshold.append(means)

    threshold = np.array(threshold)
    threshold = np.reshape(threshold,(intermediate_output.shape[0],intermediate_output.shape[3]))
    '''
    
    ASA = [] #stress activation threshold

    for k in range(intermediate_output.shape[3]): # iterate channel in batch,height,width,channel
        ASA_per_channel = []
        for i in range(intermediate_output.shape[0]): #iterate batch 
            ASA_per_channel.append(np.mean(intermediate_output[i,:,:,k]))
        ASA_per_channel = np.array(ASA_per_channel)
        ASA.append(np.mean(ASA_per_channel)) 

    ASA = np.array(ASA)
    ASA = np.mean(ASA) + 3*np.std(ASA) #the threshold
    #print("SA threshold of the given reference images is:",ASA)
    


    Auv = intermediate_layer_model.predict(target_image) #intermediate output of interest of img
    #print(Auv.shape)

    FI = []
    for i in range(intermediate_output.shape[3]): #per channel
        deltaAuv = Auv[:,:,:,i]-ASA#[i]
        Iuv = deltaAuv.copy() #indicator function check the subtracted feature map whether its positive or negative per pixel
        Iuv[Iuv <= 0] = 0
        Iuv[Iuv > 0] = 1
        if np.sum(Iuv) != 0:
            FeatureImportanceMetric = np.sum(Iuv*deltaAuv)/np.sum(Iuv)
        else:
            FeatureImportanceMetric = 0
        FI.append(FeatureImportanceMetric)
        #break
    FI = np.array(FI) #final feature importance metric
    #print("FI is:",FI,FI.shape)
        
    explanationperimage= []
    #get top3 feature indxs
    #print("Auv shape is ",Auv.shape)
    indxs = np.argsort(-FI)[:3]
    for i in indxs:
        deltaAuv = Auv[0,:,:,i]-ASA#[i]
        Iuv = deltaAuv.copy()
        Iuv[Iuv <= 0] = 0
        Iuv[Iuv > 0] = 1
        if np.sum(Iuv)==0:
            break
        FeatureImportanceMetric = np.sum(Iuv*deltaAuv) / np.sum(Iuv)
        explanationperimage.append(FeatureImportanceMetric*Iuv*deltaAuv)
    EMuv=np.array(explanationperimage)
    #print("EMuv shape is",EMuv.shape)
    
    EMuvs=np.zeros((Auv.shape[1],Auv.shape[2])) #height and width

    for i in range(EMuv.shape[0]):
        EMuvs += EMuv[i]
    
    EMuvs = zoom(EMuvs,target_image.shape[1]/EMuvs.shape[0])
    
    del intermediate_layer_model #for repetative analysis
    
    return EMuvs