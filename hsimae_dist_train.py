from common.engine_pertrain import train_one_epoch, test_one_epoch
from common.datautils import GF5Dataset
from common.dist_utils import init_distributed_mode, is_main_process
from torch.utils.data import DataLoader
from models.HsiMAE import hsimae_15p_204c_sstiny_model_64
import torch
# from torch.utils.data.dataset import ConcatDataset
import math
from common.tools import BatchSchedulerDistributedSampler, DistConcatDataset
import os
import argparse
import torch.distributed as dist
import tempfile
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter


def main(args):
    init_distributed_mode(args=args)
    device = torch.device(args.device)

    if args.rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        args.model_archive = f'runs/exp1'
        if os.path.exists('runs'):
            n = max([int(i[-1]) for i in os.listdir('runs')]) + 1
            # n = int(os.listdir('runs')[-1][-1]) + 1
            args.model_archive = f'runs/exp{n}'
        writer = SummaryWriter(args.model_archive)

    tr_root = args.tr_root
    tr_list = os.listdir(tr_root)
    tr_dataset_list = [GF5Dataset(img_root=os.path.join(tr_root, path)) for path in tr_list]
    te_root = args.te_root
    te_list = os.listdir(te_root)
    te_dataset_list = [GF5Dataset(img_root=os.path.join(te_root, path)) for path in te_list]
    tr_concat_dataset = DistConcatDataset(tr_dataset_list)
    te_concat_dataset = DistConcatDataset(te_dataset_list)

    tr_sampler = BatchSchedulerDistributedSampler(dataset=tr_concat_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_replicas=args.world_size, rank=args.rank)
    te_sampler = BatchSchedulerDistributedSampler(dataset=te_concat_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_replicas=args.world_size, rank=args.rank)

    tr_dataloader = torch.utils.data.DataLoader(dataset=tr_concat_dataset,
                                                sampler=tr_sampler,
                                                batch_size=args.batch_size // args.world_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=args.num_workers)

    te_dataloader = torch.utils.data.DataLoader(dataset=te_concat_dataset,
                                                sampler=te_sampler,
                                                batch_size=args.batch_size // args.world_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=args.num_workers)

    model_without_ddp = hsimae_15p_204c_sstiny_model_64.to(device)

    if os.path.exists(args.weights):
        print('load', args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model_without_ddp.state_dict()[k].numel() == v.numel()}
        model_without_ddp.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if args.rank == 0:
            torch.save(model_without_ddp.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model_without_ddp.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # DDP mode
    model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu])

    # optimizer
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(pg, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        tr_sampler.set_epoch(epoch)
        tr_mean_loss, tr_mean_loss_r, tr_mean_loss_c = train_one_epoch(model, optimizer, tr_dataloader, device, epoch, args.epochs)
        scheduler.step()
        te_mean_loss, te_mean_loss_r, te_mean_loss_c = test_one_epoch(model, te_dataloader, device, epoch, args.epochs)

        if is_main_process():
            #  write_tb
            writer.add_scalar('Train/loss', tr_mean_loss, epoch)
            writer.add_scalar('Valid/loss', te_mean_loss, epoch)

            writer.add_scalar('Train/loss_r', tr_mean_loss_r, epoch)
            writer.add_scalar('Valid/loss_r', te_mean_loss_r, epoch)

            writer.add_scalar('Train/loss_c', tr_mean_loss_c, epoch)
            writer.add_scalar('Valid/loss_c', te_mean_loss_c, epoch)

            # writer.add_scalar('Metric/PSNR', mean_psnr, epoch)
            # writer.add_scalar('Metric/SSIM', mean_ssim, epoch)

            # save model
            if te_mean_loss < best_loss:
                best_loss = te_mean_loss
                torch.save(model.module.state_dict(), os.path.join(args.model_archive, 'best.pth'))

            torch.save(model.module.state_dict(), os.path.join(args.model_archive, 'last.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data configuration
    parser.add_argument('--tr_root', type=str, default=r'../data/GF5_patches/train')
    parser.add_argument('--te_root', type=str, default=r'../data/GF5_patches/test')
    parser.add_argument('--num_workers', type=int, default=8)

    # train configuration
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=360)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    # dist  configuration
    parser.add_argument('--model_archive', default='runs/')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--init_method', type=str, default='env://')
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--rank', type=int, default=-1)

    args = parser.parse_args()

    main(args)
