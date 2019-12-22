from keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, Embedding, Activation, Concatenate, Conv1D, MaxPooling1D, \
    Flatten, MaxPooling2D, Multiply, Permute, Reshape, RepeatVector, Dot, Lambda, Dropout, Layer

from keras.optimizers import Adam
from keras import backend as K
from itertools import product
import numpy as np


def embedlayer(inputNodes, layername, x_idx, MAX_LENGTH=0):
    """Create embedding layer for a feature"""
    x_input = Input(shape=(MAX_LENGTH,), name=layername, dtype='int32')
    inputNodes.append(x_input)
    emb_size = int(min(len(x_idx)/2, 5))
    x_layer = Embedding(output_dim=emb_size, input_dim=len(x_idx), mask_zero=True)(x_input)
    x_layer = Dropout(0.2)(x_layer)

    return inputNodes, x_layer


def extralayer(inputNodes, x, numnode, extraidx, featuresidx, MAX_LENGTH):
    """Create embedding layer for extra feature for argument detection"""
    posidx, neridx, depidx, distanceidx, chnkidx, wikineridx, dbpedianeridx, subneridx = featuresidx


    eventidx, positionidx, rootparseidx, yesnoidx = extraidx

    inputNodes, nearevent_layer = embedlayer(inputNodes, "nearevent_input", eventidx, MAX_LENGTH)
    x = Concatenate()([x, nearevent_layer])
    numnode += len(eventidx)

    inputNodes, distfromtrigger_layer = embedlayer(inputNodes, "distfromtrigger_input", distanceidx, MAX_LENGTH)
    x = Concatenate()([x, distfromtrigger_layer])
    numnode += len(distanceidx)

    inputNodes, triggerposition_layer = embedlayer(inputNodes, "triggerposition_input", positionidx, MAX_LENGTH)
    x = Concatenate()([x, triggerposition_layer])
    numnode += len(positionidx)


    inputNodes, deppathtotrigger0_layer = embedlayer(inputNodes, "deppathtotrigger0_input", depidx, MAX_LENGTH)
    x = Concatenate()([x, deppathtotrigger0_layer])
    numnode += len(depidx)


    inputNodes, deppathtotrigger1_layer = embedlayer(inputNodes, "deppathtotrigger1_input", depidx, MAX_LENGTH)
    x = Concatenate()([x, deppathtotrigger1_layer])
    numnode += len(depidx)

    inputNodes, deppathtotrigger2_layer = embedlayer(inputNodes, "deppathtotrigger2_input", depidx, MAX_LENGTH)
    x = Concatenate()([x, deppathtotrigger2_layer])
    numnode += len(depidx)

    inputNodes, deppathtotrigger3_layer = embedlayer(inputNodes, "deppathtotrigger3_input", depidx, MAX_LENGTH)
    x = Concatenate()([x, deppathtotrigger3_layer])
    numnode += len(depidx)

    inputNodes, deppathtotrigger4_layer = embedlayer(inputNodes, "deppathtotrigger4_input", depidx, MAX_LENGTH)
    x = Concatenate()([x, deppathtotrigger4_layer])
    numnode += len(depidx)

    inputNodes, deppathtotriggerlength_layer = embedlayer(inputNodes, "deppathtotriggerlength_input", distanceidx,MAX_LENGTH)
    x = Concatenate()([x, deppathtotriggerlength_layer])
    numnode += len(distanceidx)

    inputNodes, commonrootwtrigger_layer = embedlayer(inputNodes, "commonrootwtriggerparse_input", rootparseidx,MAX_LENGTH)
    x = Concatenate()([x, commonrootwtrigger_layer])
    numnode += len(rootparseidx)

    inputNodes, depthcommonroot_layer = embedlayer(inputNodes, "depthofcommonrootwtrigger_input", distanceidx, MAX_LENGTH)
    x = Concatenate()([x, depthcommonroot_layer])
    numnode += len(distanceidx)

    inputNodes, isonly1_isnearest_layer = embedlayer(inputNodes, "isonly1_isnearest_input", yesnoidx, MAX_LENGTH)
    x = Concatenate()([x, isonly1_isnearest_layer])
    numnode += len(yesnoidx)

    return inputNodes, x, numnode

class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape

def to_categorical_sentence(sequences, categories):
    """Change label to one hot encoded, one sentence one label"""
    cat_sequences = []
    for s in sequences:
        cat_sequences.append(np.zeros(categories))
        cat_sequences[-1][s] = 1.0
    return np.array(cat_sequences)

def to_categorical_word(sequences, categories):
    """ change label to list of one hot encoded, set 1 to its labelidx, one word one label
        Input:  sequences-list of label
              categories-list of labelidx
        Output: array of one hot encoded
    """
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)

    return np.array(cat_sequences)


class WeightedCategoricalCrossEntropy(object):
  """This class is from Keras github repo"""
  def __init__(self, weights):
    nb_cl = len(weights)
    self.weights = np.ones((nb_cl, nb_cl))
    for class_idx, class_weight in weights.items():
      self.weights[0][class_idx] = class_weight
      self.weights[class_idx][0] = class_weight
    self.__name__ = 'w_categorical_crossentropy'

  def __call__(self, y_true, y_pred):
    return self.w_categorical_crossentropy(y_true, y_pred)

  def w_categorical_crossentropy(self, y_true, y_pred):
    nb_cl = len(self.weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max, axis=-1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        w = K.cast(self.weights[c_t, c_p], K.floatx())
        y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
        y_t = K.cast(y_true[..., c_t], K.floatx())
        final_mask += w * y_p * y_t
    return K.categorical_crossentropy(y_true,y_pred) * final_mask