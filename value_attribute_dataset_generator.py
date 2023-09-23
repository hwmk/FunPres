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
import itertools
import math
from torch.utils.data import DataLoader


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


parser = argparse.ArgumentParser(description='Dataset Arguments')
parser.add_argument('--values', type=int, default=10,
                    help='Number of values')
parser.add_argument('--attributes', type=int, default=4,
                    help='Number of attributes')
parser.add_argument('--train_scale', type=int, default=1,
                    help='Scaling factor for train data')
parser.add_argument('--test_scale', type=int, default=1,
                    help='Scaling factor for test data')
parser.add_argument('--test_percentage', type=float, default=0.2,
                    help='Percentage of testing data')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')


args = parser.parse_args()


iters = [range(args.values)
        for _ in range(args.attributes)]
examples = list(itertools.product(*iters))
random.shuffle(examples)

train_percentage = 1 - args.test_percentage
example_combinations = examples[:int(len(examples)*train_percentage)]
print(f'Train data length:- {len(example_combinations)}')
train_data = AbstractDataset(example_combinations, args.values, args.attributes, scale = (math.floor(args.batch_size/len(example_combinations)) + 1))

example_combinations = examples[int(len(examples)*train_percentage):]
print(f'Test data length:- {len(example_combinations)}')
test_data = AbstractDataset(example_combinations, args.values, args.attributes, scale = (math.floor(args.batch_size/len(example_combinations)) + 1))

train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
