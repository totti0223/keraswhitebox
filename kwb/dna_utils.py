import sys
import re
from shapely.wkt import loads as load_wkt
from shapely import affinity
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import time,math
#from progressbar import ProgressBar
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from random import choice, randint

#https://github.com/kundajelab/dragonn/blob/master/examples/workshop_tutorial.ipynb
A_data = """
MULTIPOLYGON (
((24.7631 57.3346, 34.3963 57.3346, 52.391 -1.422, 44.1555 -1.422, 39.8363
  13.8905, 19.2476 13.8905, 15.0039 -1.422, 6.781 -1.422, 24.7631 57.3346)),
((29.5608 50.3205, 21.1742 20.2623, 37.9474 20.2623, 29.5608 50.3205))
)
"""

C_data = """POLYGON((
52.391 2.5937, 48.5882 0.8417, 44.68 -0.4142, 40.5998 -1.17, 36.2814 -1.422,
32.8755 -1.2671, 29.6656 -0.8024, 26.6518 -0.0278, 23.834 1.0565,
21.2122 2.4507, 18.7865 4.1547, 16.5569 6.1686, 14.5233 8.4922,
12.7087 11.0966, 11.136 13.9527, 9.8053 17.0606, 8.7166 20.4201,
7.8698 24.0314, 7.2649 27.8943, 6.902 32.009, 6.781 36.3754, 6.9027 40.7209,
7.2678 44.8198, 7.8764 48.6722, 8.7283 52.278, 9.8236 55.6371,
11.1624 58.7497, 12.7446 61.6157, 14.5702 64.2351, 16.6133 66.5753,
18.8481 68.6034, 21.2745 70.3195, 23.8926 71.7235, 26.7023 72.8156,
29.7037 73.5956, 32.8967 74.0637, 36.2814 74.2197, 40.5998 73.9697,
44.68 73.2196, 48.5882 71.9696, 52.391 70.2196, 52.391 60.1101,
48.6468 62.739, 44.6331 64.657, 40.4709 65.8289, 36.2814 66.2196,
31.7716 65.7557, 29.7437 65.1758, 27.8672 64.3641, 26.1421 63.3203,
24.5684 62.0447, 23.146 60.5371, 21.875 58.7976, 19.7831 54.6129,
18.289 49.481, 17.3925 43.4019, 17.0936 36.3754, 17.3925 29.3763,
18.289 23.3166, 19.7831 18.1964, 21.875 14.0157, 23.146 12.2762,
24.5684 10.7686, 26.1421 9.4929, 27.8672 8.4492, 29.7437 7.6375,
31.7716 7.0576, 36.2814 6.5937, 40.5354 6.9844, 44.7034 8.1563,
48.6878 10.0743, 52.391 12.7032, 52.391 2.5937))"""

G_data = """POLYGON((
52.391 5.4974, 50.49 3.8964, 48.4724 2.502, 46.3383 1.3144, 44.0877 0.3334,
41.7314 -0.4346, 39.2805 -0.9832, 34.0946 -1.422, 30.9504 -1.2772,
27.9859 -0.843, 25.2009 -0.1191, 22.5956 0.8942, 20.1698 2.197,
17.9236 3.7894, 15.857 5.6713, 13.9699 7.8428, 12.285 10.2753,
10.8248 12.9404, 9.5892 15.8381, 8.5782 18.9685, 7.7919 22.3315,
7.2303 25.9271, 6.8933 29.7553, 6.781 33.8161, 6.8948 37.8674,
7.2362 41.6888, 7.8053 45.2803, 8.6019 48.6419, 9.6262 51.7737, 10.878 54.6755,
12.3575 57.3474, 14.0646 59.7895, 15.9743 61.9712, 18.0615 63.862,
20.3262 65.4618, 22.7685 66.7708, 25.3884 67.789, 28.1857 68.5162,
31.1606 68.9525, 34.3131 69.098, 38.5048 68.7957, 42.5144 67.8889,
46.3638 66.3703, 50.0748 64.2325, 50.0748 54.8075, 46.342 57.8466,
42.5144 59.9716, 38.5266 61.2226, 34.3131 61.6395, 30.1132 61.2053,
28.2228 60.6624, 26.4723 59.9024, 24.8614 58.9253, 23.3904 57.731,
22.0591 56.3195, 20.8675 54.691, 18.9046 50.7806, 17.5025 45.998,
16.6612 40.3432, 16.3808 33.8161, 16.6526 27.1962, 17.4679 21.4959,
18.8267 16.7151, 20.7291 12.8539, 21.8892 11.2595, 23.1951 9.8776,
24.6469 8.7084, 26.2446 7.7517, 27.9883 7.0076, 29.8778 6.4762, 34.0946 6.051,
36.9534 6.2276, 39.4407 6.7575, 41.6331 7.6625, 43.607 8.9644, 43.607 27.2172,
33.7304 27.2172, 33.7304 34.7776, 52.391 34.7776, 52.391 5.4974
))"""

T_data = """POLYGON((
6.781 58.3746, 52.391 58.3746, 52.391 51.5569, 33.6933 51.5569, 33.6933 -1.422,
25.5684 -1.422, 25.5684 51.5569, 6.781 51.5569, 6.781 58.3746
))"""

def create_simulated_sequence(no):
    p_sequences = []
    n_sequences = []
    #p=ProgressBar(maxval=no).start()
    for i in range(no):
        sequence = dna_sequence_generator(500)
        sequence = embed_motif(sequence,"AAATATCT",randint(0,500-9))
        p_sequences.append(sequence)

        sequence = dna_sequence_generator(500)
        while sequence.find('AAATATCT')>-1:
            sequence = dna_sequence_generator(500)
        n_sequences.append(sequence)
        #p.update(i+1)
        #time.sleep(0.01)
    p_sequences = np.array(p_sequences)
    n_sequences = np.array(n_sequences)
    dense_sequences = np.r_[p_sequences,n_sequences]
    sequences = one_hot(dense_sequences)
    sequences = sequences.astype('float32')
    labels = []
    for i in range(no):
        labels.append(np.array([1, 0]))
    for i in range(no):
        labels.append(np.array([0, 1]))
    labels = np.array(labels)
    labels = labels.astype('float32')

    names =[]
    for i in range(no*2):
        names.append(i)
    names = np.array(names)
    return sequences, labels, names

def dna_sequence_generator(length):
    def weightedchoice(items):  # this doesn't require the numbers to add up to 100
        return choice("".join(x * y for x, y in items))
    #rice gc 43.54%
    DNA = ""
    for i in range(length):
        DNA += weightedchoice([("C", 22), ("G", 22), ("A", 28), ("T", 28)])
    return DNA
def embed_motif(sequence, cis, position):
    embeded = sequence[:position] + cis + sequence[position + 6:]
    return embeded

def one_hot(sequences):
    '''
    https://github.com/kundajelab/dragonn/blob/master/dragonn/utils.py
    '''
    sequence_length = len(sequences[0])
    integer_type = np.int8 if sys.version_info[0] == 2 else np.int32  # depends on Python version
    integer_array = LabelEncoder().fit(np.array(('ACGTNRMSWYK',)).view(integer_type)).transform(sequences.view(integer_type)).reshape(len(sequences), sequence_length)
    #integer_array = LabelEncoder().fit(np.array(('ACGTN',)).view(integer_type)).transform(sequences.view(integer_type)).reshape(len(sequences), sequence_length)
    one_hot_encoding = OneHotEncoder(sparse=False, n_values=11, dtype=integer_type).fit_transform(integer_array)
    one_hot_encoding = one_hot_encoding.reshape(len(sequences), 1, sequence_length, 11)[:,:,:,[0,1,2,8]] # a picture like numpy array
   
    return one_hot_encoding
    #return one_hot_encoding.reshape(len(sequences), 1, sequence_length, 5).swapaxes(2, 3)[:, :, [0, 1, 2, 4], :]

def dense_to_one_hot(labels_dense, num_classes):
    #https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L56
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    #print (index_offset,labels_one_hot)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
    
def dict2np_whole(genelist,shuffle=True):
    '''
    takes dictionary, shuffles, and retrieves label and sequence.
    if ref == True, randomly samples from non used genes.
    '''
    sequences = []
    expression = []
    names = []

    if shuffle == True:
        idxs = np.arange(0, len(genelist))
        np.random.shuffle(idxs)
        shuffled_lists = genelist[idxs]
        #print (shuffled_lists)
        names = shuffled_lists[:,0]     
        sequences = shuffled_lists[:,1]
        expressions = shuffled_lists[:,2:]
    else:
        names = genelist[:,0]     
        sequences = genelist[:,1]
        expressions = genelist[:,2:]
    return np.array(sequences), np.array(expressions).astype(np.float), np.array(names)

def process_dataset(dictionary,shuffle=False):
    sequences, expressions, names = dict2np_whole(dictionary,shuffle=shuffle)

    print ("One hot enconding sequence.")
    sequences = one_hot(sequences)
    
    #log2 transform expression. and round it. mask 0
    #expressions[expressions==0]=0.1
    #print ("log10 transorming expressions")
    #expressions = np.log10(expressions)
    #expressions = np.log2(expressions)/np.log2(100)
    #expressions[expressions<0]=0
    mexpression = np.max(expressions.flatten())
    print ("max expression level is, ",mexpression)
    minexpression = np.min(expressions.flatten())
    print ("min expression level is, ",minexpression)
    #print ("scaling to 0 to 1")
    #expressions /= mexpression
    return sequences, expressions,names

class Polygon(object):
    # Adapt Shapely or GeoJSON/geo_interface polygons to a common interface

    def __init__(self, context):
        if hasattr(context, 'interiors'):
            self.context = context
        else:
            self.context = getattr(context, '__geo_interface__', context)

    @property
    def geom_type(self):
        return (
            getattr(self.context, 'geom_type', None) or self.context['type'])

    @property
    def exterior(self):
        return (
            getattr(self.context, 'exterior', None) or self.context['coordinates'][0])

    @property
    def interiors(self):
        value = getattr(self.context, 'interiors', None)
        if value is None:
            value = self.context['coordinates'][1:]
        return value
def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""
    this = Polygon(polygon)
    assert this.geom_type == 'Polygon'

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = np.ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals
    vertices = np.concatenate(
        [np.asarray(this.exterior)] + [np.asarray(r)
                                       for r in this.interiors])
    codes = np.concatenate(
        [coding(this.exterior)] + [coding(r)
                                   for r in this.interiors])
    return Path(vertices, codes)
def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object
    The `polygon` may be a Shapely
    or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.
    Example (using Shapely Point and a matplotlib axes):
      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)
    """
    return PathPatch(PolygonPath(polygon), **kwargs)
def standardize_polygons_str(data_str):
    """Given a POLYGON string, standardize the coordinates to a 1x1 grid.
    Input : data_str (taken from above)
    Output: tuple of polygon objects
    """
    # find all of the polygons in the letter (for instance an A
    # needs to be constructed from 2 polygons)
    path_strs = re.findall("\(\(([^\)]+?)\)\)", data_str.strip())

    # convert the data into a numpy array
    polygons_data = []
    for path_str in path_strs:
        data = np.array([
            tuple(map(float, x.split())) for x in path_str.strip().split(",")])
        polygons_data.append(data)

    # standardize the coordinates
    min_coords = np.vstack(data.min(0) for data in polygons_data).min(0)
    max_coords = np.vstack(data.max(0) for data in polygons_data).max(0)
    for data in polygons_data:
        data[:, ] -= min_coords
        data[:, ] /= (max_coords - min_coords)

    polygons = []
    for data in polygons_data:
        polygons.append(load_wkt(
            "POLYGON((%s))" % ",".join(" ".join(map(str, x)) for x in data)))

    return tuple(polygons)

letters_polygons = {}
letters_polygons['A'] = standardize_polygons_str(A_data)
letters_polygons['C'] = standardize_polygons_str(C_data)
letters_polygons['G'] = standardize_polygons_str(G_data)
letters_polygons['T'] = standardize_polygons_str(T_data)
colors = dict(zip(
    'ACGT', (('green', 'white'), ('blue',), ('orange',), ('red',))
))


def add_letter_to_axis(ax, let, x, y, height):
    """Add 'let' with position x,y and height height to matplotlib axis 'ax'.
    """
    for polygon, color in zip(letters_polygons[let], colors[let]):
        new_polygon = affinity.scale(
            polygon, yfact=height, origin=(0, 0, 0))
        new_polygon = affinity.translate(
            new_polygon, xoff=x, yoff=y)
        patch = PolygonPatch(
            new_polygon, edgecolor=color, facecolor=color)
        ax.add_patch(patch)
    return
def add_letters_to_axis(ax, letter_heights):
    """
    Plots letter on user-specified axis.
    Parameters
    ----------
    ax : axis
    letter_heights: Nx4 array
    """
    assert letter_heights.shape[1] == 4

    x_range = [1, letter_heights.shape[0]]
    pos_heights = np.copy(letter_heights)
    pos_heights[letter_heights < 0] = 0
    neg_heights = np.copy(letter_heights)
    neg_heights[letter_heights > 0] = 0

    for x_pos, heights in enumerate(letter_heights):
        #print ("x_pos",x_pos)
        #print ("heights",heights)
        letters_and_heights = sorted(zip(heights, 'ACGT'))
        #print ("letters and heights",letters_and_heights)
        y_pos_pos = 0.0
        y_neg_pos = 0.0
        for height, letter in letters_and_heights:
            #print ("height is",height)
            if height > 0:
                add_letter_to_axis(ax, letter, 0.5 + x_pos, y_pos_pos, height)
                y_pos_pos += height
            else:
                add_letter_to_axis(ax, letter, 0.5 + x_pos, y_neg_pos, height)
                y_neg_pos += height

    ax.set_xlim(x_range[0] - 1, x_range[1] + 1)
    ax.set_xticks(list(range(*x_range)) + list([x_range[-1]]))
    ax.set_aspect(aspect='auto', adjustable='box')
    ax.autoscale_view()
    
def plot_sequence_filters(conv_filters):
    fig = plt.figure(figsize=(25, 4))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    num_plots_per_axis = int(len(conv_filters)**0.5) + 1
    for i, conv_filter in enumerate(conv_filters):
        #print (conv_filter)
        ax = fig.add_subplot(num_plots_per_axis, num_plots_per_axis, i+1)
        add_letters_to_axis(ax, conv_filter)
        ax.axis("off")
        #ax.set_title("Filter %s" % (str(i+1)))

#print ("conv1",conv1[0][0].T[0])

#fig = plt.figure(figsize=(15, 8))
#fig.subplots_adjust(hspace=0.1, wspace=0.1)
#ax = fig.add_subplot(1,1,1)
#print (conv1)
