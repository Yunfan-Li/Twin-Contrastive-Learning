import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist


def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg


def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = (
        torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    )
    precision = (
        1 + torch.arange(k, device=logits.device).float()
    ) / labels_to_sorted_idx
    return precision.sum(1) / k


class InstanceLoss(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, tau=0.5, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = z / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        return loss


class ClusterLoss(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, tau=1.0, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

    def forward(self, c, get_map=False):
        n = c.shape[0]
        assert n % self.multiplier == 0

        # c = c / np.sqrt(self.tau)

        if self.distributed:
            c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            c_list = diffdist.functional.all_gather(c_list, c)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            c_list = [chunk for x in c_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            c_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    c_sorted.append(c_list[i * self.multiplier + m])
            c_aug0 = torch.cat(
                c_sorted[: int(self.multiplier * dist.get_world_size() / 2)], dim=0
            )
            c_aug1 = torch.cat(
                c_sorted[int(self.multiplier * dist.get_world_size() / 2) :], dim=0
            )

            p_i = c_aug0.sum(0).view(-1)
            p_i /= p_i.sum()
            en_i = np.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
            p_j = c_aug1.sum(0).view(-1)
            p_j /= p_j.sum()
            en_j = np.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
            en_loss = en_i + en_j

            c = torch.cat((c_aug0.t(), c_aug1.t()), dim=0)
            n = c.shape[0]

        c = F.normalize(c, p=2, dim=1) / np.sqrt(self.tau)

        logits = c @ c.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        return loss + en_loss


class InstanceLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(
        self,
        tau=0.5,
        multiplier=2,
        distributed=False,
        alpha=0.99,
        gamma=0.5,
        cluster_num=10,
    ):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.alpha = alpha
        self.gamma = gamma
        self.cluster_num = cluster_num

    @torch.no_grad()
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        if self.distributed:
            c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            pseudo_label_cur_list = [torch.zeros_like(pseudo_label_cur) for _ in range(dist.get_world_size())]
            index_list = [torch.zeros_like(index) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            c_list = diffdist.functional.all_gather(c_list, c)
            pseudo_label_cur_list = diffdist.functional.all_gather(pseudo_label_cur_list, pseudo_label_cur)
            index_list = diffdist.functional.all_gather(index_list, index)
            c = torch.cat(c_list, dim=0,)
            pseudo_label_cur = torch.cat(pseudo_label_cur_list, dim=0,)
            index = torch.cat(index_list, dim=0,)
        batch_size = c.shape[0]
        device = c.device
        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
        tmp = torch.arange(0, batch_size).to(device)

        prediction = c.argmax(dim=1)
        confidence = c.max(dim=1).values
        unconfident_pred_index = confidence < self.alpha
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(
            int
        )
        for i in range(self.cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        return pseudo_label_nxt.cpu(), index

    def forward(self, z, pseudo_label):
        n = z.shape[0]
        assert n % self.multiplier == 0

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            pseudo_label_list = [
                torch.zeros_like(pseudo_label) for _ in range(dist.get_world_size())
            ]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            pseudo_label_list = diffdist.functional.all_gather(
                pseudo_label_list, pseudo_label
            )
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            pseudo_label_list = [
                chunk for x in pseudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
                    pesudo_label_sorted.append(
                        pseudo_label_list[i * self.multiplier + m]
                    )
            z_i = torch.cat(
                z_sorted[: int(self.multiplier * dist.get_world_size() / 2)], dim=0
            )
            z_j = torch.cat(
                z_sorted[int(self.multiplier * dist.get_world_size() / 2) :], dim=0
            )
            pseudo_label = torch.cat(pesudo_label_sorted, dim=0,)
            n = z_i.shape[0]

        invalid_index = pseudo_label == -1
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(
            z_i.device
        )
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(z_i.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()

        contrast_count = self.multiplier
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.tau
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        mask = mask.repeat(anchor_count, contrast_count)
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n * anchor_count).view(-1, 1).to(z_i.device),
            0,
        )
        logits_mask *= 1 - mask
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, n).mean()

        return instance_loss


class ClusterLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, multiplier=1, distributed=False, cluster_num=10):
        super().__init__()
        self.multiplier = multiplier
        self.distributed = distributed
        self.cluster_num = cluster_num

    def forward(self, c, pseudo_label):
        if self.distributed:
            # c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            pesudo_label_list = [
                torch.zeros_like(pseudo_label) for _ in range(dist.get_world_size())
            ]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # c_list = diffdist.functional.all_gather(c_list, c)
            pesudo_label_list = diffdist.functional.all_gather(
                pesudo_label_list, pseudo_label
            )
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            # c_list = [chunk for x in c_list for chunk in x.chunk(self.multiplier)]
            pesudo_label_list = [
                chunk for x in pesudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            # c_sorted = []
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    # c_sorted.append(c_list[i * self.multiplier + m])
                    pesudo_label_sorted.append(
                        pesudo_label_list[i * self.multiplier + m]
                    )
            # c = torch.cat(c_sorted, dim=0)
            pesudo_label_all = torch.cat(pesudo_label_sorted, dim=0)
        pseudo_index = pesudo_label_all != -1
        pesudo_label_all = pesudo_label_all[pseudo_index]
        idx, counts = torch.unique(pesudo_label_all, return_counts=True)
        freq = pesudo_label_all.shape[0] / counts.float()
        weight = torch.ones(self.cluster_num).to(c.device)
        weight[idx] = freq
        pseudo_index = pseudo_label != -1
        if pseudo_index.sum() > 0:
            criterion = nn.CrossEntropyLoss(weight=weight).to(c.device)
            loss_ce = criterion(
                c[pseudo_index], pseudo_label[pseudo_index].to(c.device)
            )
        else:
            loss_ce = torch.tensor(0.0, requires_grad=True).to(c.device)
        return loss_ce
