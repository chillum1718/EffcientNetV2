import argparse
import csv
import os

import torch
import tqdm
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from nets import nn
from utils import util

data_dir = os.path.join('..', 'Dataset', 'IMAGENET')


def batch(images, target, model, criterion=None):
    images = images.cuda()
    target = target.cuda()
    if criterion:
        with torch.cuda.amp.autocast():
            loss = criterion(model(images), target)
        return loss
    else:
        return util.accuracy(model(images), target, top_k=(1, 5))


def train(args):
    epochs = 350
    batch_size = 288
    util.set_seeds(args.rank)
    model = nn.EfficientNet().cuda()
    lr = batch_size * torch.cuda.device_count() * 0.256 / 4096
    optimizer = nn.RMSprop(util.add_weight_decay(model), lr, 0.9, 1e-3, momentum=0.9)
    ema = nn.EMA(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                   transforms.Compose([util.RandomResize(),
                                                       transforms.ColorJitter(0.4, 0.4, 0.4),
                                                       transforms.RandomHorizontalFlip(),
                                                       util.RandomAugment(),
                                                       transforms.ToTensor(), normalize]))
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    loader = data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    scheduler = nn.StepLR(optimizer)
    amp_scale = torch.cuda.amp.GradScaler()
    with open(f'weights/{scheduler.__str__()}.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'acc@1', 'acc@5'])
            writer.writeheader()
        best_acc1 = 0
        for epoch in range(0, epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                bar = tqdm.tqdm(loader, total=len(loader))
            else:
                bar = loader
            model.train()
            for images, target in bar:
                loss = batch(images, target, model, criterion)
                optimizer.zero_grad()
                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                ema.update(model)
                torch.cuda.synchronize()
                if args.local_rank == 0:
                    bar.set_description(('%10s' + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), loss))

            scheduler.step(epoch + 1)
            if args.local_rank == 0:
                acc1, acc5 = test(ema.model.eval())
                writer.writerow({'acc@1': str(f'{acc1:.3f}'),
                                 'acc@5': str(f'{acc5:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                util.save_checkpoint({'state_dict': ema.model.state_dict()}, acc1 > best_acc1)
                best_acc1 = max(acc1, best_acc1)
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(model=None):
    if model is None:
        model = nn.EfficientNet()
        model.load_state_dict(torch.load('weights/best.pt', 'cpu')['state_dict'])
        model = model.cuda()
        model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                   transforms.Compose([transforms.Resize(416),
                                                       transforms.CenterCrop(384),
                                                       transforms.ToTensor(), normalize]))

    loader = data.DataLoader(dataset, 48, num_workers=os.cpu_count(), pin_memory=True)
    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    with torch.no_grad():
        for images, target in tqdm.tqdm(loader, ('%10s' * 2) % ('acc@1', 'acc@5')):
            acc1, acc5 = batch(images, target, model)
            torch.cuda.synchronize()
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        acc1, acc5 = top1.avg, top5.avg
        print('%10.3g' * 2 % (acc1, acc5))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return acc1, acc5


def print_parameters():
    model = nn.EfficientNet().eval()
    _ = model(torch.zeros(1, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {int(params)}')


def benchmark():
    shape = (1, 3, 384, 384)
    util.torch2onnx(nn.EfficientNet().export().eval(), shape)
    util.onnx2caffe()
    util.print_benchmark(shape)


def main():
    # python -m torch.distributed.launch --nproc_per_node=3 main.py --train
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    args.distributed = False
    args.rank = 0
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.rank = torch.distributed.get_rank()
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    if args.local_rank == 0:
        print_parameters()
    if args.benchmark:
        benchmark()
    if args.train:
        train(args)
    if args.test:
        test()


if __name__ == '__main__':
    main()
