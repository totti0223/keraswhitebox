import numpy as np
from scipy.ndimage.interpolation import zoom

def patch_occlusion(model,
                    images,
                    substitute_values=0,
                    patch_size=10,
                    strides=5,
                    normalize=True):
    '''
    Args:
        model (keras model)
        images: shape (height,width,channel) or (batch,height,width,channel)
        substitute vales: values to be substituted.
        patch_size: size to be occluded. defaults to be square.
    Returns:
        numpy arrays of occulusion prediction results:(batch,height,width) or (height,width)
        Note: if input image has batch dimention, output will also harbor batch dimention
    '''    
    #lmodel = linearize_activation(model)
    if len(images.shape) == 3:
        images = np.expand_dims(images,axis=0)
    B = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    C = images.shape[3]
    vals = []
    val = []
    #infer per image
    for batch in range(B): #iterate per image
        substituted = []
        base = model.predict(np.expand_dims(images[batch],axis=0))[0]
        cls = np.argmax(base)
        base_val = np.max(base)
        occluded_images = []
        height_of_occluded = 0
        width_of_occluded = 0

        for h in range(H)[::strides]:
            height_of_occluded += 1
            for w in range(W)[::strides]:
                width_of_occluded += 1
                tmp = images[batch].copy() #1,2000,4
                
                hstart = np.maximum(h-int(patch_size/2),0)
                hend = np.minimum(h+int(patch_size/2),tmp.shape[0])
                wstart = np.maximum(w-int(patch_size/2),0)
                wend = np.minimum(w+int(patch_size/2),tmp.shape[1])
                #print("w",w)
                #print(wstart,wend)
                #print(wend-wstart)
                replace_val = np.full((hend-hstart,wend-wstart,3),substitute_values)
                
                tmp[hstart:hend,wstart:wend] = replace_val
                #print(h-int(patch_size/2),h+int(patch_size/2))
                occluded_images.append(tmp)
                #val.append(model.predict(np.expand_dims(tmp,axis=0))[0,cls])
        width_of_occluded /= height_of_occluded
        width_of_occluded = int(width_of_occluded)
        occluded_images = np.array(occluded_images)
        val = model.predict(occluded_images,batch_size=10)[:,cls]

        if normalize:
            #normalize by subtracting the default inference value
            val -= base_val
            #negative values means important so swaping pos neg
            val *= -1
            val = np.maximum(0,val)
            val /= val.max()
        val = np.reshape(val,(height_of_occluded,width_of_occluded))
        #val = resize(val,(H,W))
        val = zoom(val,H/val.shape[0],mode="reflect")
        vals.append(val)
    
    vals = np.array(vals)
    vals = np.reshape(vals,(B,H,W))
    if vals.shape[0] == 1:
        vals = vals[0]
    return vals

def pixel_wise_occlusion(model, images,substitute_values=-0.25,skip_pixel=16,method="normalize"):
    '''
    occlusion analysis is finding the important pixels within the image by
    creating numbers of images with each pixel nullyfied and check how the
    output changes. if 224*224 images, 224*224 images have to be prepared
    and analyzed so it is very costly and not preferred. so sacrificing the
    final output resolution, we will occulude every XX pixel by the argv 
    skip_pixel and forcely put back to the original resolution by resizing.
    another way of occulusion is enlarging the occuluding area (ex. 4x4) 
    but makes the output shape complicated not writing the codes for now.
    preferred to use patch_occlusion for now
    
    Args:
        model (keras model)
        images: shape (height,width,channel) or (batch,height,width,channel)
        substitute vales: values to be substituted.
        skip_pixel: how many pixels you want to skip per occulusion. must be evenly divided by height and width, respectively
    Returns:
        numpy arrays of occulusion prediction results:(batch,height,width) or (height,width)
        Note: if input image has batch dimention, output will also harbor batch dimention
    
    '''

    if len(images.shape) == 3:
        images = np.expand_dims(images,axis=0)
    B = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    C = images.shape[3]
    vals = []
    val = []
    #infer per image
    for batch in range(B): #iterate per image
        substituted = []
        base = model.predict(np.expand_dims(images[batch],axis=0))[0]
        cls = np.argmax(base)
        base_val = np.max(base)
        for h in range(H)[::skip_pixel]:
            for w in range(W)[::skip_pixel]:
                tmp = images[batch].copy() #1,2000,4
                replace_val = np.array([substitute_values]*C)
                tmp[h,w] = replace_val
                val.append(model.predict(np.expand_dims(tmp,axis=0))[0,cls])
                
        val = np.array(val)
        
        if method=="normalize":
            #normalize by subtracting the default inference value
            val -= base_val
            #negative values means important so swaping pos neg
            val *= -1
            val = np.maximum(0,val)
            val /= val.max()
        val = np.reshape(val,(int(H/skip_pixel),int(W/skip_pixel)))
        #val = resize(val,(H,W))
        val = zoom(val,H/val.shape[0])
        vals.append(val)
    
    vals = np.array(vals)
    vals = np.reshape(vals,(B,H,W))
    if vals.shape[0] == 1:
        vals = vals[0]
    return vals