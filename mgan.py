import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):
        pass

class Generator(nn.Module):

    def __init__(self):
        """
        Generator component of GAN. requires an input slightly bigger 
        than 300 x 300 (precisely 306 x 306 (?))
        """
        super(Generator, self).__init__()
       
        # 5 x 5 square convolution.
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


if __name__ == "__main__":
    generator = Generator()
