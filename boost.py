import argparse
import time
import datetime
import misc
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from data import build_dataset
from model import get_resnet, Network
from misc import NativeScalerWithGradNormCount as NativeScaler
from loss import InstanceLossBoost, ClusterLossBoost
from engine import boost_one_epoch, evaluate
import json


def get_args_parser():
    parser = argparse.ArgumentParser("TCL", add_help=False)
    parser.add_argument(
        "--batch_size", default=256, type=int, help="Batch size per GPU"
    )
    parser.add_argument("--epochs", default=200, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="resnet34",
        type=str,
        metavar="MODEL",
        choices=["resnet50", "resnet34", "resnet18"],
        help="Name of model to train",
    )
    parser.add_argument("--feat_dim", default=128, type=int, help="dimension of ICH")
    parser.add_argument(
        "--ins_temp",
        default=0.5,
        type=float,
        help="temperature of instance-level contrastive loss",
    )
    parser.add_argument(
        "--clu_temp",
        default=1.0,
        type=float,
        help="temperature of cluster-level contrastive loss",
    )

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (absolute lr)",
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", default="./datasets/", type=str, help="dataset path",
    )
    parser.add_argument(
        "--dataset",
        default="CIFAR-10",
        type=str,
        help="dataset",
        choices=["CIFAR-10", "CIFAR-100", "ImageNet-10", "ImageNet"],
    )
    parser.add_argument(
        "--nb_cluster", default=10, type=int, help="number of the clusters",
    )
    parser.add_argument(
        "--output_dir",
        default="./save/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--resume",
        default="./save/checkpoint-0.pth",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--save_freq", default=20, type=int, help="saving frequency")
    parser.add_argument(
        "--eval_freq", default=10, type=int, help="evaluation frequency"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(type="train", args=args)
    dataset_pseudo = build_dataset(type="pseudo", args=args)
    dataset_val = build_dataset(type="val", args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_pseudo = torch.utils.data.DistributedSampler(
        dataset_pseudo, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_ps = torch.utils.data.DataLoader(
        dataset_pseudo,
        sampler=sampler_pseudo,
        batch_size=1000,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    backbone, hidden_dim = get_resnet(args)
    model = Network(backbone, hidden_dim, args.feat_dim, args.nb_cluster)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint["model"]

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    feat_vector = []
    labels_vector = []
    for (images, labels, _) in metric_logger.log_every(data_loader_val, 20, header):
        images = images.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            feat, c = model.forward_zc(images)
            c = torch.argmax(c, dim=1)

        feat_vector.extend(feat.cpu().detach().numpy())
        labels_vector.extend(labels.numpy())
    feat_vector = np.array(feat_vector)
    labels_vector = np.array(labels_vector)
    print(
        "Feat shape {}, Label shape {}".format(feat_vector.shape, labels_vector.shape)
    )

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * misc.get_world_size()

    print("base lr: %.3e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = torch.optim.Adam(
        [
            {"params": model.resnet.parameters(), "lr": args.lr,},
            {"params": model.instance_projector.parameters(), "lr": args.lr},
            {"params": model.cluster_projector.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    loss_scaler = NativeScaler()

    criterion_ins = InstanceLossBoost(
        tau=args.ins_temp, distributed=True, alpha=0.99, gamma=0.5
    )
    criterion_clu = ClusterLossBoost(distributed=True, cluster_num=args.nb_cluster)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    pseudo_labels = -torch.ones(dataset_train.__len__(), dtype=torch.long)
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats, pseudo_labels = boost_one_epoch(
            model,
            criterion_ins,
            criterion_clu,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            pseudo_labels,
            args=args,
        )
        if args.output_dir and (
            epoch % args.save_freq == 0 or epoch + 1 == args.epochs
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        if (
            epoch % args.eval_freq == 0
            or epoch + 1 == args.epochs
        ):
            test_stats = evaluate(data_loader_val, model, device)
            print(
                f"Clustering performance on the {len(dataset_val)} test images: NMI={test_stats['nmi']:.2f}%, ACC={test_stats['acc']:.2f}%, ARI={test_stats['ari']:.2f}%"
            )
            max_accuracy = max(max_accuracy, test_stats["acc"])
            print(f"Max accuracy: {max_accuracy:.2f}%")
        
        if epoch == args.start_epoch:
            test_stats = {"pred_num": 1000}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
