# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys

import torch
import torch.nn.functional as F

import misc
import numpy as np
from evaluate import cluster_metric


def train_one_epoch(
    model,
    criterion_ins,
    criterion_clu,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    args,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for data_iter_step, ((x_w, x_s, x), _, index) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            z_i, z_j, c_i, c_j = model(x_w, x_s)
            c_i = F.softmax(c_i, dim=1)
            c_j = F.softmax(c_j, dim=1)
            loss_ins = criterion_ins(torch.concat((z_i, z_j), dim=0))
            loss_clu = criterion_clu(torch.concat((c_i, c_j), dim=0))
            loss = loss_ins + loss_clu

        loss_ins_value = loss_ins.item()
        loss_clu_value = loss_clu.item()

        if not math.isfinite(loss_ins_value) or not math.isfinite(loss_clu_value):
            print(
                "Loss is {}, {}, stopping training".format(
                    loss_ins_value, loss_clu_value
                )
            )
            sys.exit(1)

        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=True,
        )
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_ins=loss_ins_value)
        metric_logger.update(loss_clu=loss_clu_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    pred_vector = []
    labels_vector = []
    for (images, labels, _) in metric_logger.log_every(data_loader, 20, header):
        images = images.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            preds = model.module.forward_c(images)
            preds = torch.argmax(preds, dim=1)

        pred_vector.extend(preds.cpu().detach().numpy())
        labels_vector.extend(labels.numpy())
    pred_vector = np.array(pred_vector)
    labels_vector = np.array(labels_vector)
    print(
        "Pred shape {}, Label shape {}".format(pred_vector.shape, labels_vector.shape)
    )

    nmi, ari, acc = cluster_metric(labels_vector, pred_vector)
    print(nmi, ari, acc)

    metric_logger.meters["nmi"].update(nmi)
    metric_logger.meters["acc"].update(acc)
    metric_logger.meters["ari"].update(ari)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def boost_one_epoch(
    model,
    criterion_ins,
    criterion_clu,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    pseudo_labels,
    args,
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()
    for data_iter_step, ((x_w, x_s, x), _, index) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)

        model.eval()
        with torch.cuda.amp.autocast(), torch.no_grad():
            _, _, c = model(x, x, return_ci=False)
            c = F.softmax(c / args.clu_temp, dim=1)
            pseudo_labels_cur, index_cur = criterion_ins.generate_pseudo_labels(
                c, pseudo_labels[index].to(c.device), index.to(c.device)
            )
            pseudo_labels[index_cur] = pseudo_labels_cur
            pseudo_index = pseudo_labels != -1
            metric_logger.update(pseudo_num=pseudo_index.sum().item())
            metric_logger.update(
                pseudo_cluster=torch.unique(pseudo_labels[pseudo_index]).shape[0]
            )
        if epoch == args.start_epoch:
            continue

        model.train(True)
        with torch.cuda.amp.autocast():
            z_i, z_j, c_j = model(x_w, x_s, return_ci=False)
            loss_ins = criterion_ins(
                torch.concat((z_i, z_j), dim=0), pseudo_labels[index].to(x_s.device)
            )
            loss_clu = criterion_clu(c_j, pseudo_labels[index].to(x_s.device))
            loss = loss_ins + loss_clu

        loss_ins_value = loss_ins.item()
        loss_clu_value = loss_clu.item()

        if not math.isfinite(loss_ins_value) or not math.isfinite(loss_clu_value):
            print(
                "Loss is {}, {}, stopping training".format(
                    loss_ins_value, loss_clu_value
                )
            )
            sys.exit(1)
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=True,
        )
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_ins=loss_ins_value)
        metric_logger.update(loss_clu=loss_clu_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        pseudo_labels,
    )

