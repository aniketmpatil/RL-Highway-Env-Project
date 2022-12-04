from keras.models import Sequential
from keras.layers import Flatten, Permute

def Identity(obs_shape):
    model = Sequential()
    model.add(Permute((3,2,1), input_shape=obs_shape))
    model.add(Flatten())
    return model

model_factory = {"Identity":Identity}
def get_model(opt):
    return model_factory[opt.arch](opt.obs_dim)