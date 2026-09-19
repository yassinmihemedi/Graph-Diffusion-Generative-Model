import numpy as np
import argparse
from diffusion.utils import add_parent_path, set_seeds, clean_args

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Exp
from experiment import GraphExperiment, add_exp_args

# Model
from model import get_model, get_model_id, add_model_args

# Optim
from diffusion.optim.multistep import get_optim, get_optim_id, add_optim_args

###########
## Setup ##
###########


parser = argparse.ArgumentParser()
add_data_args(parser)
add_exp_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = clean_args(parser.parse_args())

# DDP: init process group before anything uses args.device so that
# each process trains on its own GPU.
if args.parallel == 'ddp':
    import os, torch.distributed as dist
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    args.device = f'cuda:{local_rank}'

set_seeds(args.seed)

##################
## Specify data ##
##################

train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics, train_density = get_data(args)

args.num_edge_classes = num_edge_classes
args.num_node_classes = num_node_classes

args.has_node_feature = False   # default: no typed node features
if args.final_prob_node is None:
    args.final_prob_node = [1-1e-12, 1e-12]
    args.num_node_classes = 2
else:
    # final_prob_node was set by get_data() (e.g. ZINC atom type marginal):
    # the model should output node class logits.
    args.has_node_feature = True

if 0 in args.final_prob_edge:
    args.final_prob_edge[np.argmax(args.final_prob_edge)] = args.final_prob_edge[np.argmax(args.final_prob_edge)]-1e-12
    args.final_prob_edge[np.argmin(args.final_prob_edge)] = 1e-12

# DiGress-style marginal transition: if the user left --final_prob_edge at its
# untouched ~empty-graph default, point the forward process's t→∞ target at the
# dataset's true edge density instead. An explicit --final_prob_edge still wins.
if args.final_prob_edge == [1 - 1e-12, 1e-12] and train_density is not None:
    args.final_prob_edge = [1.0 - train_density, train_density]
args.train_density = train_density

args.max_degree = max_degree
args.num_node_feat = num_node_feat
args.augmented_feature_dict = augmented_feature_dict



data_id = get_data_id(args)
###################
## Specify model ##
###################

model = get_model(args, initial_graph_sampler=initial_graph_sampler)
#print('model', model)
model_id = get_model_id(args)
#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)

##############
## Training ##
##############
exp = GraphExperiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=eval_loader,
                 test_loader=test_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch,
                 monitoring_statistics=monitoring_statistics,
                 eval_evaluator=eval_evaluator, 
                 test_evaluator=test_evaluator,
                 n_patient=5000)

exp.run()
