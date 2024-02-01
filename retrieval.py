import argparse
import datetime
import json
import logging
import os
import random
import time
import numpy as np
import yaml
import torch
from torch import distributed, optim
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import transforms
from dataset import create_loader, create_sampler, get_dataset
from evaluation import evaluation, itm_eval
from unire.model import unire

import utils
from scheduler import create_scheduler
from optim import create_optimizer

from sentence_transformers import SentenceTransformer


def main(args, config):
    device = torch.device(args.gpu)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if args.resume:
        # try to continue training
        print('load checkpoint from %s' % args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        best_epoch = checkpoint['best_epoch']
        # config = checkpoint['config']
    else:
        start_epoch = 0
        best = 0
        best_epoch = 0
        state_dict = None

    print("args: ", args)
    print("config: ", config)
    print("config prefix: ", json.dumps(config, indent=4))

    # for training
    # get model
    # when resume, state_dict is not None, so we can load model from state_dict
    print("Creating model")
    model = unire(args, config)
    msg = model.load_state_dict(state_dict)
    print(msg)
    model.to(device)

    # get dataset
    print("Creating dataset")
    if args.experiment:
        train_dataset, val_dataset, test_dataset = [get_dataset(config['dataset_name'], config['data_path'], split, model.preprocess) for split in [
            'experiment', 'val', "test"]]
    else:
        train_dataset, val_dataset, test_dataset = [get_dataset(
            config['dataset_name'], config['data_path'], split, model.preprocess) for split in ['train', 'val', "test"]]
    # get sampler
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    # get loader
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers, batch_size=[config['batch_size_train'], config[
        'batch_size_test'], config['batch_size_testall']], num_workers=[16, 16, 16], is_trains=[True, False, False], collate_fns=[None, None, None])

    # get distributed model
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # assisant model
    # use sentence transformer to get text softlabel
    txt_enc_assisant = SentenceTransformer('all-mpnet-base-v2').to(device=device)
    if args.distributed:
        txt_enc_assisant = torch.nn.parallel.DistributedDataParallel(txt_enc_assisant, device_ids=[args.gpu])

    # train setting
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    # optimizer
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    # scheduler
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    #  train
    print("Start training")
    start_time = time.time()

    if args.eval:
        print("Start eval")
        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, args)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, args)
        if utils.is_main_process():
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            print(val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(test_result)
        # synchronize()
        dist.barrier()
        # release gpu memory
        torch.cuda.empty_cache()
        return

    for epoch in range(start_epoch, max_epoch):
        lr_scheduler.step(epoch)
        # set epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_stats = {}
        # train
        train_stats = train(model, train_loader, optimizer, lr_scheduler, epoch, warmup_steps, device, config, txt_enc_assisant)

        # eval
        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, args)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, args)

        # save model and log
        if utils.is_main_process():
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            print(val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(test_result)
            print("Train stats:", train_stats)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_result.items()},
                         **{f'test_{k}': v for k, v in test_result.items()},
                         'epoch': epoch,
                         }
            with open(os.path.join(config['logger_name'], "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if test_result['r_mean'] > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'best': best,
                    'best_epoch': best_epoch,
                }
                torch.save(save_obj, os.path.join(config['model_name'], 'checkpoint_best.pth'))
                best = test_result['r_mean']
                best_epoch = epoch

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'config': config,
                'epoch': epoch,
                'best': best,
                'best_epoch': best_epoch,
            }
            torch.save(save_obj, os.path.join(
                config['model_name'], 'checkpoint_{}.pth'.format(str(epoch).zfill(2))))

        # synchronize()
        dist.barrier()
        # release gpu memory
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(config['logger_name'], "log.txt"), "a") as f:
            f.write("best epoch: %d\n\n" % best_epoch)


def train(model, train_loader, optimizer, lr_scheduler, epoch, warmup_steps, device, config, txt_enc_assisant):
    model.train()

    # set metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_contrastive', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cross_modal', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_uni_modal', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metrics = [
        "tau",
        "cross_tau", "cross_tau_image", "cross_tau_text", "cross_the_softlabel_tau", "cross_the_softlabel_tau_image", "cross_the_softlabel_tau_text",
        "uni_tau", "uni_tau_image", "uni_tau_text", "uni_the_softlabel_tau", "uni_the_softlabel_tau_image", "uni_the_softlabel_tau_text"
    ]
    for val in metrics:
        if hasattr(model.module, val):
            metric_logger.add_meter(val, utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, image_features, raw_captions, idx) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)
        caption = caption.to(device, non_blocking=True)

        # softlabel feature for cross-modal retrieval and uni-modal retrieval
        with torch.no_grad():
            image_features = image_features.to(device, non_blocking=True)
            caption_features = txt_enc_assisant.module.encode(
                raw_captions, device=device, show_progress_bar=False, convert_to_tensor=True).to(device, non_blocking=True)
        # get loss
        cross_modal_loss, uni_modal_loss, contrastive_loss = model(image, caption, image_features, caption_features, epoch, idx)

        loss = cross_modal_loss + uni_modal_loss + contrastive_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric logger
        for val in metrics:
            if hasattr(model.module, val):
                metric_logger.update(**{val: getattr(model.module, val).item()})
        metric_logger.update(loss_cross_modal=cross_modal_loss.item())
        metric_logger.update(loss_uni_modal=uni_modal_loss.item())
        metric_logger.update(loss_contrastive=contrastive_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def parser_args():
    parser = argparse.ArgumentParser(description="PyTorch Image Retrieval Training")
    parser.add_argument('--config', type=str, default='', help='The config file.')
    parser.add_argument('--eval', action='store_true', help='Is eval?')
    parser.add_argument('--experiment', action='store_true', help='Is experiment?')
    parser.add_argument('--resume', action='store_true', help='Is resume?')
    parser.add_argument('--seed', default=23, type=int, help='Seed for initializing training.')
    parser.add_argument("--num_workers", default=8, type=int, help="The number of workers to use for data loading.")
    parser.add_argument('--distributed', default=True, type=bool, help='Is distributed?')
    parser.add_argument('--checkpoint', type=str, default='', help='The checkpoint file to resume from.')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set env
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # set args
    args = parser_args()
    # set distributed
    utils.init_distributed_mode(args)

    assert not (args.config == '' and args.checkpoint == ''), "config and checkpoint cannot be empty at the same time"
    config = None
    if args.config != '':
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config['save_path'] = config['save_path'] + "_seed" + str(args.seed)
            config['logger_name'] = os.path.join(config['save_path'], "log")
            config['model_name'] = os.path.join(config['save_path'], "checkpoints")

    if args.resume and args.checkpoint == '':
        modelList = os.listdir(config['model_name'])
        modelList.sort()
        modelPath = modelList[-2]
        args.checkpoint = os.path.join(config['model_name'], modelPath)

    if utils.is_main_process():
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])
        # Copy the configuration file to storage
        try:
            # If the file exists
            if os.path.exists(args.config):
                os.system("cp -f %s %s" % (args.config, os.path.join(config['save_path'])+"/"))
        except:
            pass
        if not os.path.exists(config['model_name']):
            os.makedirs(config['model_name'])
        if not os.path.exists(config['logger_name']):
            os.makedirs(config['logger_name'])
    main(args, config)
