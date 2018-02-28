import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
from PIL import Image

class Pokemon(Dataset):
    """ Pokemon data (loaded from data directory) """


    def __init__(self):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.pokemon = [Image.open(g) for g in glob.glob("data/*.png")]
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pokemon)

    def __getitem__(self, idx):
        return self.transform(self.pokemon[idx]), torch.ones(1)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # Almost the same as the one in the tutorial
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(82944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # Almost the same as the one in the tutorial
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self._num_flat_features(x))
        print(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

def train(epochs, d_steps, g_steps):
    pokemon_dataset = Pokemon()
    pokemon_loader = torch.utils.data.DataLoader(pokemon_dataset,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=4)
    discriminator = Discriminator()
    generator = Generator()
    optimizer = torch.optim.Adam(discriminator.parameters())
    for epoch in range(epochs):
        for (inputs, targets) in pokemon_loader:
            # Train the discriminator.
            inputs = Variable(inputs)
            targets = Variable(targets)

            result = discriminator(inputs)
            loss = F.mse_loss(result, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for g_index in range(d_steps):
            # Train the generator.
            pass

def main():
    train(1000, 20, 100)


if __name__ == "__main__":
    main()
