import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets.ModelNetDataset import ModelNet40SVM
from models.ResNet import ResNet
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # trainval
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])
        losses_pc = AverageMeter(['Loss_pc'])
        losses_fea = AverageMeter(['Loss_fea'])

        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (pc, img) in enumerate(train_dataloader):
            pc, img = pc.cuda(), img.cuda()
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints

            assert pc.size(1) == npoints
            pc = train_transforms(pc)
            loss_pc, loss_fea = base_model(pc, img)
            loss = loss_pc + loss_fea
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item() * 1000])
            losses_pc.update([loss_pc.item() * 1000])
            losses_fea.update([loss_fea.item() * 1000])

            lr = optimizer.param_groups[0]['lr']
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_pc', loss_pc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_fea', loss_fea.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', lr, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f(s) DataTime = %.3f(s) Losses = low_pc: %s, high_fea: %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                             ['%.4f' % l for l in losses_pc.val()], ['%.4f' % l for l in losses_fea.val()], lr), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_pc', losses_pc.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_fea', losses_fea.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s low_pc: %s, high_fea: %s, lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], ['%.4f' % l for l in losses_pc.avg()],
             ['%.4f' % l for l in losses_fea.avg()], optimizer.param_groups[0]['lr']), logger = logger)

        # extra_train_dataloader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=64, shuffle=True)
        # test_dataloader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=64, shuffle=False)
        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
        #
        #     # Save ckeckpoints
        #     if metrics.better_than(best_metrics):
        #         best_metrics = metrics
        #         builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 10 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (data, label) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (data, label) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass