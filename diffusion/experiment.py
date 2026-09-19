import torch
import torch.distributed as dist
from diffusion.utils import get_args_table, clean_dict, str_to_bool

# Path
import os
import time
import pathlib
HOME = str(pathlib.Path.home())

# Experiment
from diffusion import BaseExperiment
from diffusion.base import DataParallelDistribution, DDPDistribution

#  Logging frameworks
from torch.utils.tensorboard import SummaryWriter
import wandb


def add_exp_args(parser):

    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp', 'ddp'})
    parser.add_argument('--resume', type=str, default=None)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--log_tb', type=str_to_bool, default=True)
    parser.add_argument('--log_wandb', type=str_to_bool, default=True)
    parser.add_argument('--log_home', type=str, default='./wandb')


class DiffusionExperiment(BaseExperiment):
    no_log_keys = ['project', 'name',
                   'log_tb', 'log_wandb',
                   'check_every', 'eval_every',
                   'device', 'parallel'
                   'pin_memory', 'num_workers']

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, eval_loader, test_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch, 
                 monitoring_statistics, n_patient, eval_evaluator, test_evaluator):
        if args.log_home is None:
            self.log_base = os.path.join(HOME, 'log', 'flow')
        else:
            self.log_base = args.log_home

        # Edit args
        if args.eval_every is None:
            args.eval_every = args.epochs
        if args.check_every is None:
            args.check_every = args.epochs
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        # Move model
        model = model.to(args.device)
        self._train_sampler = None
        if args.parallel == 'dp':
            model = DataParallelDistribution(model)
        elif args.parallel == 'ddp':
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            model = DDPDistribution(model, device_ids=[local_rank],
                                    find_unused_parameters=True)
            # Replace train_loader's sampler with DistributedSampler
            from torch.utils.data.distributed import DistributedSampler
            from torch.utils.data import DataLoader
            self._train_sampler = DistributedSampler(train_loader.dataset)
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=self._train_sampler,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
                collate_fn=train_loader.collate_fn,
            )

        # Init parent
        super(DiffusionExperiment, self).__init__(model=model,
                                                  optimizer=optimizer,
                                                  scheduler_iter=scheduler_iter,
                                                  scheduler_epoch=scheduler_epoch,
                                                  log_path=os.path.join(self.log_base, data_id, model_id, optim_id, args.name),
                                                  eval_every=args.eval_every,
                                                  check_every=args.check_every,
                                                  monitoring_statistics=monitoring_statistics,
                                                  n_patient=n_patient, 
                                                  eval_evaluator=eval_evaluator, 
                                                  test_evaluator= test_evaluator)

        # Store args (must be set before create_folders so is_main_process works)
        self.args = args
        self.create_folders()
        self.save_args(args)

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        
        # Init logging (rank 0 only for DDP)
        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        if self.is_main_process:
            if args.log_tb:
                self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
                self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)
            if args.log_wandb:
                wandb.init(config=args_dict, project=args.project, id=args.name, dir=self.log_path)

    @property
    def is_main_process(self):
        return self.args.parallel != 'ddp' or dist.get_rank() == 0

    def create_folders(self):
        if self.is_main_process:
            super(DiffusionExperiment, self).create_folders()

    def save_args(self, args):
        if self.is_main_process:
            super(DiffusionExperiment, self).save_args(args)

    def save_metrics(self):
        if self.is_main_process:
            super(DiffusionExperiment, self).save_metrics()

    def checkpoint_save(self, name='checkpoint.pt'):
        if self.is_main_process:
            # Save the underlying module's state dict so checkpoints are
            # portable (no 'module.' prefix from DDP wrapper).
            if self.args.parallel == 'ddp':
                import pickle
                checkpoint = {
                    'current_epoch': self.current_epoch,
                    'train_metrics': self.train_metrics,
                    'eval_metrics': self.eval_metrics,
                    'test_metrics': self.test_metrics,
                    'eval_epochs': self.eval_epochs,
                    'model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                    'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None,
                }
                torch.save(checkpoint, os.path.join(self.check_path, name))
            else:
                super(DiffusionExperiment, self).checkpoint_save(name)

    def run(self):
        if self.args.resume:
            self.resume()
        for epoch in range(self.current_epoch, self.args.epochs):
            # DDP: set epoch on sampler so shuffling differs each epoch
            if self._train_sampler is not None:
                self._train_sampler.set_epoch(epoch)

            train_dict = self.train_fn(epoch)
            self.log_train_metrics(train_dict)

            if (epoch + 1) % self.eval_every == 0:
                eval_dict = self.eval_fn(epoch)
                self.log_eval_metrics(eval_dict)
                self.eval_epochs.append(epoch)
                if self.compare_current_best(eval_dict):
                    self.current_best_eval_dict = eval_dict
                    test_dict = self.test_fn(epoch)
                    self.log_test_metrics(test_dict)
                    self.test_epochs.append(epoch)
                    self.patient = 0
                else:
                    self.patient += 1
            else:
                eval_dict = None
                test_dict = None

            self.save_metrics()
            self.log_fn(epoch, train_dict, eval_dict, test_dict)

            self.current_epoch += 1
            if (epoch + 1) % self.check_every == 0:
                self.checkpoint_save(f'checkpoint_{epoch}.pt')

            if self.patient > self.n_patient:
                return

    def log_fn(self, epoch, train_dict, eval_dict, test_dict):
        if not self.is_main_process:
            return

        # Tensorboard
        if self.args.log_tb:
            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if test_dict:
                for metric_name, metric_value in test_dict.items():
                    self.writer.add_scalar('test/{}'.format(metric_name), metric_value, global_step=epoch+1)

        # Weights & Biases — single log call per epoch so all metrics land at
        # the same step atomically; avoids partial logging if the job is killed
        # mid-loop or W&B has a transient network hiccup.
        if self.args.log_wandb:
            wb_dict = {f'base/{k}': v for k, v in train_dict.items()}
            if eval_dict:
                wb_dict.update({f'eval/{k}': v for k, v in eval_dict.items()})
            if test_dict:
                wb_dict.update({f'test/{k}': v for k, v in test_dict.items()})
            wandb.log(wb_dict, step=epoch+1)

    def resume(self):
        resume_path = os.path.join(self.log_base, self.data_id, self.model_id, self.optim_id, self.args.resume, 'check')
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]
            if epoch in self.eval_epochs:
                eval_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    eval_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else: 
                eval_dict = None
            
            if epoch in self.test_epochs:
                test_dict = {}
                for metric_name, metric_values in self.test_metrics.items():
                    test_dict[metric_name] = metric_values[self.test_epochs.index(epoch)]
            else: 
                test_dict = None
            self.log_fn(epoch, train_dict=train_dict, eval_dict=eval_dict, test_dict=test_dict)

