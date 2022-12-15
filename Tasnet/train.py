import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.pit_wrapper as module_loss
import model.sdr as module_func_loss
from model.sdr import multisrc_neg_sisdr
import model.metric as module_metric
#import model.model as module_arch
from utility import conv_tasnet as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import wandb
import os


# is this needed?!
# wandb.init(project="my-AV-project", entity="ren_mor")
# wandb.run.name = config['trainer']['run_name']

# sweep_id = wandb.sweep(sweep_config)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    wandb.init(project="TasNet-baseline", entity="ren_mor")
    wandb.run.name = os.path.basename(config["trainer"]["save_dir"])
    wandb.config = config
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    print("passsss model")
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    func_loss = getattr(module_func_loss, config["loss"]["loss_func"])
    reduce = False
    if config["loss"]["perm_reduce"] is not False:
        reduce = getattr(module_loss, config["loss"]["perm_reduce"])
    else:
        reduce=None
    kw = {"loss_func": func_loss, "perm_reduce": reduce}
    #criterion = multisrc_neg_sisdr
    criterion = config.init_obj('loss', module_loss, **kw)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    #wandb.log({"loss": func_loss})
    # Optional
    wandb.watch(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    print(type(trainable_params))
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    num_of_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters that require grad in the model is: {num}".format(num=num_of_param))
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="/home/lab/renana/PycharmProjects/Conv-TasNet/Conv-TasNet-master/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options, trainer_or_tester="trainer",  save_path="save_dir")
    main(config)
    print("made change in VS code")
