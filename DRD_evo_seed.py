import numpy as np
import math
from copy import deepcopy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_grad_vector, get_future_step_parameters
from scipy.spatial.distance import pdist, squareform
from torch.distributions import  MultivariateNormal
import random

dist_kl = lambda y, t_s : F.kl_div(F.log_softmax(y, dim=-1), F.softmax(t_s, dim=-1), reduction='mean') * y.size(0)
entropy_fn = lambda x : torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1), dim=-1)
cross_entropy = lambda y, t_s : -torch.sum(F.log_softmax(y, dim=-1)*F.softmax(t_s, dim=-1),dim=-1).mean()
mse = torch.nn.MSELoss()

def WGF_evolve_update(args, model, opt, input_x, input_y, evolve=True):

    if evolve:
        orig_x, orig_y= deepcopy(input_x), deepcopy(input_y)
        z_hat = deepcopy(orig_x)
        z_hat = z_hat.clone().detach().requires_grad_(True)

        for n in range(args.T_adv):
            delta = z_hat - orig_x
            rho = torch.mean((torch.norm(delta.view(len(orig_x), -1), 2, 1) ** 2))
            loss_zt = F.cross_entropy(model(z_hat), orig_y)
            loss_phi = - (loss_zt - args.gamma * rho)
            loss_phi.backward()
            target_grad = z_hat.grad
            deltanorm = torch.norm(delta, 2)

            if args.method == 'DRD-LD':
                z_hat = LD_step(args.stepsize, z_hat, target_grad)
            elif args.method == 'DRD-GD':
                input_shape = z_hat.size()
                flat_z = z_hat.view(input_shape[0], -1)
                target_grad = target_grad.view(input_shape[0], -1)
                flat_z = GD_step(args.stepsize, flat_z, target_grad)
                z_hat = flat_z.view(list(input_shape))

            z_hat = z_hat.clone().detach().requires_grad_(True)
        opt.zero_grad()

    hid = model.return_hidden(input_x)
    logits = model.linear(hid)
    loss = F.cross_entropy(logits, input_y)

    opt.zero_grad()
    loss.backward()

    if not evolve:
        opt.step()
        return model

    logits_adv = model(z_hat)
    adv_loss = F.cross_entropy(logits_adv, orig_y)

    logits_buffer = model(orig_x)
    normal_loss = F.cross_entropy(logits_buffer, orig_y)
    total_loss = normal_loss + args.grad_ftr*adv_loss
    total_loss.backward()

    opt.step()
    return model
    
    
def LD_step(stepsize, z_gen, target_grad):
    noise_std = np.sqrt(2*stepsize)
    langevin_noise = z_gen.data.new(z_gen.data.size()).normal_(mean=0, std=noise_std)
    WGF_sample = z_gen + stepsize * target_grad + langevin_noise
    return WGF_sample


def SVGD_kernal(flat_x, h=-1):
    x_numpy = flat_x.cpu().data.numpy()
    init_dist = pdist(x_numpy)
    pairwise_dists = squareform(init_dist)
    if h < 0:
        h = np.median(pairwise_dists)
        h = 0.05 * h ** 2 / np.log(flat_x.shape[0] + 1)

    if x_numpy.shape[0] > 1:
        kernal_xj_xi = torch.exp(- torch.tensor(pairwise_dists) ** 2 / h)
    else:
        kernal_xj_xi = torch.tensor([1])
    return kernal_xj_xi, h

def GD_step(stepsize, z_gen, target_grad):
    device = target_grad.device
    kernal_xj_xi, h = SVGD_kernal(z_gen, h=-1)
    kernal_xj_xi, h = kernal_xj_xi.float(), h.astype(float)
    kernal_xj_xi = kernal_xj_xi.to(device)
    d_kernal_xi = torch.zeros(z_gen.size()).to(device)
    x = z_gen

    for i_index in range(x.size()[0]):
        d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h
    current_grad = (torch.matmul(kernal_xj_xi, target_grad) + d_kernal_xi) / x.size(0)
    WGF_sample = z_gen - stepsize * current_grad
    return WGF_sample



