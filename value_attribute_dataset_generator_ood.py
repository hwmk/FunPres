from torch.distributions.categorical import Categorical

from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    split_holdout,
    split_train_test,
)

import random
import torch
import numpy as np
import argparse


class AbstractDataset_v2:
    def __init__(self, examples, values, attributes, probs=None):
        self.examples = examples
        self.values = values
        self.attributes = attributes
        self.positions = [i for i in range(attributes)]

        if probs is None:
            self.dis = Categorical(torch.ones(attributes) / attributes)
        else:
            self.dis = Categorical(torch.tensor([probs]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, k):
        sample = torch.tensor(self.examples[k])
        mask = torch.zeros(self.attributes)

        # select number of attributes to mask
        n_masked_attributes = self.dis.sample().item()
        random.shuffle(self.positions)
        for i in range(n_masked_attributes):
            position = self.positions[i]
            mask[position] = 1

        labels_ = sample*(1-mask) - mask
        labels = torch.tensor([self.values + 1] * self.attributes)
        j = 0

        for i in range(len(labels)):
            val = labels_[i].item()
            if val > -1:
                labels[j] = val
                j = j + 1

        return sample, labels


class AbstractDataset:
    def __init__(self, examples, values, attributes, scale=False, probs=None):
        self.examples = examples
        self.attributes = attributes
        self.values = values
        if scale:
          self.scale_factor = scale
        else:
          self.scale_factor = 1

    def __len__(self):
        return len(self.examples) * self.scale_factor

    def __getitem__(self, k):
        k = k % len(self.examples)
        sample = torch.tensor(self.examples[k]) 
        mask = np.array([0,0,0,0])
        mask_ = np.array([0.1,0.1,0.1,0.1,1,1,1])           

        return sample, torch.tensor(mask), torch.tensor(mask_)


def create_tuples():
    numbers_list = list(range(2, 100))
    random_list_1 = random.sample(numbers_list, len(numbers_list))
    random_list_2 = random.sample(numbers_list, len(numbers_list))
    return [b for a,b in enumerate(zip(random_list_1, random_list_2))]

parser = argparse.ArgumentParser(description='Dataset Arguments')
parser.add_argument('--values', type=int, default=100,
                    help='Number of values')
parser.add_argument('--attributes', type=int, default=2,
                    help='Number of attributes')
parser.add_argument('--train_scale', type=int, default=1,
                    help='Scaling factor for train data')
parser.add_argument('--test_scale', type=int, default=1,
                    help='Scaling factor for test data')

args = parser.parse_args()

values = args.values
attributes = args.attributes

full_data_ood = enumerate_attribute_value(2, 100)
train_ood, generalization_holdout_ood = split_holdout(full_data_ood)
train_ood, uniform_holdout_ood = split_train_test(train_ood, 0.1)
random.shuffle(generalization_holdout_ood)
additional_training_pairs = [(0,0),(1,0),(0,1)]
train_ood = train_ood + additional_training_pairs*200

for pair in additional_training_pairs[1:]:  # (0 , 0) is not in generalization_holdout
    generalization_holdout_ood.remove(pair)

scale = args.train_scale
test_scale = args.test_scale

print(f'Values:-{values} \t Attributes:-{attributes} \t Train scale:-{scale} \t Test scale:-{test_scale}')

train_data = AbstractDataset(train_ood, 100, 2, scale=scale)
test_data = AbstractDataset(generalization_holdout_ood, 100, 2, scale=test_scale)




        
