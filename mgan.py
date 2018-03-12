import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
from PIL import Image
from itertools import islice, cycle
from functools import reduce
import operator
import numpy as np

# For truncation errors
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
            transforms.Resize((10, 10)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pokemon)

    def __getitem__(self, idx):
        return self.transform(self.pokemon[idx]), torch.ones(1)


class Population:

    def _generate_population(self):
        for rate in self.mask_rate:
            noise = torch.rand(self.shape)
            a = np.random.choice([0, 1], size=self.shape, p=[(1 - rate), rate])
            t = torch.from_numpy(a)
            t_comp = (t - 1).abs_()
            self.population.append(t.float() * noise + self.examplar * t_comp.float())

    def _create_new_population(self):
        for rate in self.mask_rate:
            # TODO: Make GPU only!
            noise = torch.rand(self.shape)
            a = np.random.choice([0, 1], self.shape, p=[(1 - rate), rate])
            t = torch.from_numpy(a) 
            self.population.append(t.float() * noise)

    def __init__(self, mask_rate, shape, examplar=None):
        """Takes an examplarly model and creates a new population from it.
           exemplar : the example "creature" that will be mutated.
           (Note: If examplar is None, then default to a "transparent image")
           mask_rate : the mutation rate, the percentage of changes for the
           populations. (As percent NOT decimal)
           shape : Shape of the resulting creature.
        """
        self.examplar = examplar
        self.mask_rate = mask_rate
        self.shape = shape
        self.population = []
        if examplar is None:
            self._create_new_population()
        else:
            self._generate_population()

    def __len__(self):
        return len(self.mask_rate)

    def __getitem__(self, index):
        return self.population[index]


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # Almost the same as the one in the tutorial
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*2*2, 32)
        self.fc2 = nn.Linear(32, 1)

    def _num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # Almost the same as the one in the tutorial
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(-1, self._num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


def to_generator(iterator):
    for _ in range(10000):
        for x in iterator:
            yield x

def train(epochs, d_steps, g_steps):
    pokemon_dataset = Pokemon()
    pokemon_loader = torch.utils.data.DataLoader(pokemon_dataset,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=4)
    pokemon_generator = to_generator(pokemon_loader)
    population = Population([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0,
                             0.0, 0.01, 0.02], (1, 4, 10, 10))
    discriminator = Discriminator()
    discriminator.cuda()
    d_optimizer = optim.SGD(discriminator.parameters(), lr=0.01)
    best = None
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        print(epoch)
        pop_index = 0
        for (inputs, targets) in islice(pokemon_generator, d_steps):
            # Train the discriminator.
            # Train on actual data.
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
            real_result = discriminator(inputs)
            print(real_result)
            #print(real_result)
            real_loss = criterion(real_result, targets)
            d_optimizer.zero_grad() 
            real_loss.backward()
            d_optimizer.step()
            
            fake_result = discriminator(Variable(population[pop_index]).cuda())
            print(fake_result[0][0])
            fake_loss = criterion(fake_result, Variable(torch.zeros(1).cuda()))
            d_optimizer.zero_grad()
            fake_loss.backward()
            d_optimizer.step()
            pop_index += 1

        best_score = None   
        while best_score is None or best_score.data[0][0] < 0.99:
            print(best_score)
            # Score the populations.
            for p in population:
                score = discriminator(Variable(p).cuda())
                if best_score is None or score.data[0][0] > best_score.data[0][0]:
                    best = p
                    best_score = score
            population = Population([0.0, 0.1, 0.2] +
                                    [0.01] * 5 +
                                    [0.02] * 5 + 
                                    [0.03] * 5 +
                                    [0.04] * 5 +
                                    [0.05] * 5, (1, 4, 10, 10), examplar=best)
            poke = best.cpu()
            poke_pil = transforms.ToPILImage()(255*poke.view(4, 10, 10))
            poke_pil.save("test.png")

            
        print(best_score[0][0])


    # Generate example image.
        poke = best.cpu()
        poke_pil = transforms.ToPILImage()(255*poke.view(4, 10, 10))
        poke_pil.save("test.png")


def main():
    train(10000, 1, 30)


if __name__ == "__main__":
    main()
