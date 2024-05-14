import numpy as np

import torch
import torch.nn as nn


def batchwise_cdist(samples1, samples2, eps=1e-6):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)

    The following broadcasting operation will be computed:
    (N x 1 x K x D) - (N x K x 1 x D) = (N x K x K x D)

    Parameters
    ----------
    samples1: torch.Tensor (shape: N x K x D)
    samples2: torch.Tensor (shape: N x K x D)

    Returns
    -------
    batchwise distance: N x K ** 2
    """
    if len(samples1.size()) != 3 or len(samples2.size()) != 3:
        raise RuntimeError('expected: 3-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')

    samples1 = samples1.unsqueeze(1)
    samples2 = samples2.unsqueeze(2)
    # Problem is distance --> Distance goes to inf
    distance = torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, -1)

    return distance


def soft_contrastive_nll(logit, matched):
    r"""Compute the negative log-likelihood of the soft contrastive loss.

    .. math::
        NLL_{ij} = -\log p(m = m_{ij} | z_i, z_j)
                 = -\log \left[ \mathbb{I}_{m_{ij} = 1} \sigma(-a \| z_i - z_j \|_2 + b)
                         +  \mathbb{I}_{m_{ij} = -1} (1 - \sigma(-a \| z_i - z_j \|_2 + b)) \right].

    Note that the matching indicator {m_ij} is 1 if i and j are matched otherwise -1.
    Here we define the sigmoid function as the following:
    .. math::
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.

    Here we sample "logit", s_{ij} by Monte-Carlo sampling to get the expected soft contrastive loss.
    .. math::
        s_{ij}^k = -a \| z_i^k - z_j^k \|_2 + b, z_i^k ~ \mathcal N (\mu_i, \Sigma_i), z_j^k ~ \mathcal N (\mu_j, \Sigma_j).

    Then we can compute NLL by logsumexp (here, we omit `k` in s_{ij}^k for the simplicity):
    .. math::
        NLL_{ij} = -\log \left[ \frac{1}{K^2} \sum_{s_{ij}} \left{ \frac{\exp(s_{ij} m_ij)}{\exp(s_{ij}) + \exp(-s_{ij})} \right} \right]
                 = (\log K^2) -\log \sum_{s_{ij}} \left[ \exp \left( s_{ij} m_ij - \log(\exp(s_{ij} + (-s_{ij}))) \right) \right]
                 = (\log K^2) -logsumexp( s_{ij} m_{ij} - logsumexp(s_{ij}, -s_{ij}) ).

    Parameters
    ----------
    logit: torch.Tensor (shape: N x K ** 2)
    matched: torch.Tensor (shape: N), an element should be either 1 (matched) or -1 (mismatched)

    Returns
    -------
    NLL loss: torch.Tensor (shape: N), should apply `reduction` operator for the backward operation.
    """
    if len(matched.size()) == 1:
        matched = matched[:, None]
    
    loss = -((logit * matched - torch.stack((logit, -logit), dim=2).logsumexp(dim=2, keepdim=False)).logsumexp(dim=1)) + np.log(logit.size(1))

    return loss


class MCSoftContrastiveLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        shift = args.shift * torch.ones(1)
        negative_scale = args.negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.uniform_lambda = args.uniform
        self.vib_beta = args.vib

        # self.shift = args.shift
        # self.negative_scale = args.negative_scale

    
    def pairwise_sampling(self, anchors, candidates, formula = None):
        N = len(anchors)
        if len(anchors) != len(candidates):
            raise RuntimeError('# anchors ({}) != # candidates ({})'.format(anchors.shape, candidates.shape))
        
        if formula != None:
            anchor_idx, selected_idx, matched = self.hard_sampling(N, formula)
        else :
            anchor_idx, selected_idx, matched = self.full_sampling(N)

        anchor_idx = torch.from_numpy(np.array(anchor_idx)).long()
        selected_idx = torch.from_numpy(np.array(selected_idx)).long()
        matched = torch.from_numpy(np.array(matched)).float()

        anchor_idx = anchor_idx.to(anchors.device)
        selected_idx = selected_idx.to(anchors.device)
        matched = matched.to(anchors.device)

        anchors = anchors[anchor_idx]
        selected = candidates[selected_idx]

        cdist = batchwise_cdist(anchors, selected)

        if cdist.isnan().sum() > 0:
            print("error")

        return cdist, matched

    def uniform_loss(self, x, max_samples=16384, t=2):
        if len(x) ** 2 > max_samples:
            # prevent CUDA error: https://github.com/pytorch/pytorch/issues/22313
            indices = np.random.choice(len(x), int(np.sqrt(max_samples)))
            x = x[indices]
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def kl_divergence(self, mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum()

    def hard_sampling(self, N, formula):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i) # 0 0 0 0 1 1 1 1
                selected.append(j)   # 0 1 2 3 0 1 2 3
                if i == j:           # 1 0 0 0 0 1 0 0 
                    matched.append(1)
                else:
                    if formula[i] == formula[j]:
                        matched.append(1)
                    else:
                        matched.append(-1)
        return candidates, selected, matched

    def full_sampling(self, N):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i) # 0 0 0 0 1 1 1 1
                selected.append(j)   # 0 1 2 3 0 1 2 3
                if i == j:           # 1 0 0 0 0 1 0 0 
                    matched.append(1)
                else:
                    matched.append(-1)
        return candidates, selected, matched

    def _compute_loss(self, input1, input2, formula = None):
        """
        Shape
        -----
        Input1 : torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Input2: torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Output: torch.Tensor
            If :attr:`reduction` is ``'none'``, then :math:`(N)`.
        """
        distance, matched = self.pairwise_sampling(input1, input2, formula)
        logits = -self.negative_scale * distance + self.shift
        self.distance = distance.mean()

        idx = matched == 1
        loss_pos = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        self.right_pos = (logits[idx] > 0).sum() / (len(matched[idx]) * distance.shape[1])
        idx = matched != 1
        loss_neg = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        self.right_neg = (logits[idx] < 0).sum() / (len(matched[idx]) * distance.shape[1])

        return {
            'loss': loss_pos + loss_neg,
            'pos_loss': loss_pos,
            'neg_loss': loss_neg,
        }
    
    def forward(self, sto, str, sto_logsigma, formula = None):
        
        uniform_loss = 0
        vib_loss = 0

        soft_con_loss = self._compute_loss(sto, str.unsqueeze(1), formula)

        if self.uniform_lambda != 0:
            dim = sto.size()[-1]
            uniform_loss = self.uniform_loss(sto.view(-1, dim))
        
        if self.vib_beta != 0:
            vib_loss = self.kl_divergence(sto.mean(dim=1), sto_logsigma)

        return soft_con_loss["loss"] + self.uniform_lambda * uniform_loss + self.vib_beta * vib_loss



class MCSoftContrastiveLoss_det(nn.Module):

    def __init__(self, args):
        super().__init__()

        shift = args.shift * torch.ones(1)
        negative_scale = args.negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.uniform_lambda = args.uniform
        self.vib_beta = args.vib

        # self.shift = args.shift
        # self.negative_scale = args.negative_scale

    
    def pairwise_sampling(self, anchors, candidates, formula = None):
        N = len(anchors)
        if len(anchors) != len(candidates):
            raise RuntimeError('# anchors ({}) != # candidates ({})'.format(anchors.shape, candidates.shape))
        
        if formula != None:
            anchor_idx, selected_idx, matched = self.hard_sampling(N, formula)
        else :
            anchor_idx, selected_idx, matched = self.full_sampling(N)

        anchor_idx = torch.from_numpy(np.array(anchor_idx)).long()
        selected_idx = torch.from_numpy(np.array(selected_idx)).long()
        matched = torch.from_numpy(np.array(matched)).float()

        anchor_idx = anchor_idx.to(anchors.device)
        selected_idx = selected_idx.to(anchors.device)
        matched = matched.to(anchors.device)

        anchors = anchors[anchor_idx]
        selected = candidates[selected_idx]

        cdist = batchwise_cdist(anchors, selected)

        if cdist.isnan().sum() > 0:
            print("error")

        return cdist, matched

    def hard_sampling(self, N, formula):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i) # 0 0 0 0 1 1 1 1
                selected.append(j)   # 0 1 2 3 0 1 2 3
                if i == j:           # 1 0 0 0 0 1 0 0 
                    matched.append(1)
                else:
                    if formula[i] == formula[j]:
                        matched.append(1)
                    else:
                        matched.append(-1)
        return candidates, selected, matched

    def full_sampling(self, N):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i) # 0 0 0 0 1 1 1 1
                selected.append(j)   # 0 1 2 3 0 1 2 3
                if i == j:           # 1 0 0 0 0 1 0 0 
                    matched.append(1)
                else:
                    matched.append(-1)
        return candidates, selected, matched

    def _compute_loss(self, input1, input2, formula = None):
        """
        Shape
        -----
        Input1 : torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Input2: torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Output: torch.Tensor
            If :attr:`reduction` is ``'none'``, then :math:`(N)`.
        """
        distance, matched = self.pairwise_sampling(input1, input2, formula)
        logits = -self.negative_scale * distance + self.shift
        self.distance = distance.mean()

        idx = matched == 1
        loss_pos = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        self.right_pos = (logits[idx] > 0).sum() / (len(matched[idx]) * distance.shape[1])
        idx = matched != 1
        loss_neg = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        self.right_neg = (logits[idx] < 0).sum() / (len(matched[idx]) * distance.shape[1])

        return {
            'loss': loss_pos + loss_neg,
            'pos_loss': loss_pos,
            'neg_loss': loss_neg,
        }
    
    def forward(self, sto, str, formula = None):

        soft_con_loss = self._compute_loss(sto, str.unsqueeze(1), formula)

        return soft_con_loss["loss"]