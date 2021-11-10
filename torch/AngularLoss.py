#! /usr/bin/env python3
# coding=utf-8

import cv2, torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter
import torch.nn.functional as F
import torchvision.models as models

def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)

class BaseLossWithValidity(object):

    def calculate_loss(self, predictions, ground_truth):
        raise NotImplementedError('Must implement BaseLossWithValidity::calculate_loss')

    def calculate_mean_loss(self, predictions, ground_truth):
        return torch.mean(self.calculate_loss(predictions, ground_truth))

    def __call__(self, predictions, gt_key, reference_dict):
        # Since we deal with sequence data, assume B x T x F (if ndim == 3)
        batch_size = predictions.shape[0]

        individual_entry_losses = []
        num_valid_entries = 0

        for b in range(batch_size):
            # Get sequence data for predictions and GT
            entry_predictions = predictions[b]
            entry_ground_truth = reference_dict[gt_key][b]

            # If validity values do not exist, return simple mean
            # NOTE: We assert for now to catch unintended errors,
            #       as we do not expect a situation where these flags do not exist.
            validity_key = gt_key + '_validity'
            assert(validity_key in reference_dict)
            # if validity_key not in reference_dict:
            #     individual_entry_losses.append(torch.mean(
            #         self.calculate_mean_loss(entry_predictions, entry_ground_truth)
            #     ))
            #     continue

            # Otherwise, we need to set invalid entries to zero
            validity = reference_dict[validity_key][b].float()
            losses = self.calculate_loss(entry_predictions, entry_ground_truth)

            # Some checks to make sure that broadcasting is not hiding errors
            # in terms of consistency in return values
            assert(validity.ndim == losses.ndim)
            assert(validity.shape[0] == losses.shape[0])

            # Make sure to scale the accumulated loss correctly
            num_valid = torch.sum(validity)
            accumulated_loss = torch.sum(validity * losses)
            if num_valid > 1:
                accumulated_loss /= num_valid
            num_valid_entries += 1
            individual_entry_losses.append(accumulated_loss)

        # Merge all loss terms to yield final single scalar
        return torch.sum(torch.stack(individual_entry_losses)) / float(num_valid_entries)

class AngularLoss(BaseLossWithValidity):

    _to_degrees = 180. / np.pi

    def calculate_loss(self, a, b):
        a = pitchyaw_to_vector(a)
        b = pitchyaw_to_vector(b)
        sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
        return torch.acos(sim) * self._to_degrees


if __name__ == "__main__":
    input = torch.randn(3, 2, requires_grad=True)
    target = torch.randn(3, 2)
    input = torch.tensor([ [1 + 2 * np.pi, 1 + 2 * np.pi] for i in range(9) ])
    target = torch.tensor([[np.pi, np.pi], [np.pi * 3 / 4, np.pi  * 3 / 4], [np.pi / 2, np.pi / 2],\
                            [np.pi / 4, np.pi / 4], [0, 0], [ - np.pi / 4, - np.pi / 4], [ - np.pi / 2, - np.pi / 2],\
                                [ - np.pi * 3 / 4, - np.pi * 3 / 4], [ - np.pi, - np.pi]])
    loss = AngularLoss()
    output = loss.calculate_loss(input, target)
    # print(target)
    # print(torch.sin(target))
    # print(torch.cos(target))
    # print(pitchyaw_to_vector(target))
    print(np.pi)
    print(output)
    print(torch.remainder(output, 2*np.pi))
    test = torch.randn(3, 2)
    sign = torch.sign(test)
    print(test)
    print(test * sign)
    print((test * sign % (2*np.pi)) * sign)


    # L1_loss = torch.nn.L1Loss(reduction='mean')
    # L2_loss = torch.nn.MSELoss()
    # input = torch.randn(3, 2, requires_grad=True)
    # target = torch.randn(3, 2)
    # output = L1_loss(input, target)
    # print(input)
    # print(target)
    # print("L1")
    # print(output)
    # print(L1_loss(input[:,0], target[:,0]))
    # print(L1_loss(input[:,1], target[:,1]))
    # output.backward()

    # output = L2_loss(input, target)
    # print("L2")
    # print(output)
    # output.backward()