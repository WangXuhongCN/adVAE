# from .alexnet import AlexNet
# from .resnet34 import ResNet34
# from .squeezenet import SqueezeNet
# from torchvision.models import InceptinV3
# from torchvision.models import alexnet as AlexNet
#from .mnist import mnist_Encoder,mnist_Decoder,mnist_Gauss_trans
from .VAE import VAE
from .AE import AE
from .self_adVAE import Encoder,Decoder,Gauss_trans #,self_adVAE
from .GAN import NetD,NetG