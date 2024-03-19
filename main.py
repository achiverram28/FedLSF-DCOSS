"""The main file"""
import argparse
from typing import Any

import numpy as np
import ray
import torch

from dataprocess import load_data
from server import Server
from train_class import Trainer_General
from utils import get_indexes, label_dirichlet_partition

ray.init()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="cora", type=str)
parser.add_argument("-c", "--global_rounds", default=100, type=int)
parser.add_argument("-i", "--local_step", default=5, type=int)

parser.add_argument("-n", "--n_trainer", default=5, type=int)
parser.add_argument("-a", "--alpha", default=0.01, type=float)
parser.add_argument("-nl", "--nlayer", default=2, type=int)
parser.add_argument("-hd", "--hidden_dim", default=32, type=int)
parser.add_argument("-nh", "--num_heads", default=2, type=int)

parser.add_argument("-td", "--tran_dropout", default=0.2, type=float)
parser.add_argument("-fd", "--feat_dropout", default=0.6, type=float)
parser.add_argument("-pd", "--prop_dropout", default=0.2, type=float)

parser.add_argument("-no", "--norm", default="none", type=str)
parser.add_argument("-lr", "--lr", default=0.05, type=float)
parser.add_argument("-wd", "--weight_decay", default=5e-3, type=float)

parser.add_argument("-r", "--repeat_time", default=3, type=int)
args = parser.parse_args()

np.random.seed(42)
torch.manual_seed(42)

adj, x, y, idx_train, idx_val, idx_test = load_data(args.dataset)

device = torch.device("cpu")

average_final_test_loss_repeats = []
average_final_test_accuracy_repeats = []


class_num = y.max().item() + 1


for repeat in range(args.repeat_time):
    split_data_indexes = label_dirichlet_partition(
        y, len(y), class_num, args.n_trainer, args.alpha
    )

    for i in range(args.n_trainer):
        split_data_indexes[i] = np.array(split_data_indexes[i])
        split_data_indexes[i].sort()
        split_data_indexes[i] = torch.tensor(split_data_indexes[i])

    in_com_train_data_indexes, in_com_test_data_indexes = get_indexes(
        split_data_indexes, args.n_trainer, idx_train, idx_test
    )

    @ray.remote(num_cpus=0.5, scheduling_strategy="SPREAD")
    class Trainer(Trainer_General):
        def __init__(self, *args: Any, **kwds: Any):
            super().__init__(*args, **kwds)

    trainers = [
        Trainer.remote(
            i,
            class_num,
            adj[split_data_indexes[i][:, None], split_data_indexes[i]],
            x[split_data_indexes[i]],
            y[split_data_indexes[i]],
            in_com_train_data_indexes[i],
            in_com_test_data_indexes[i],
            args.nlayer,
            args.hidden_dim,
            args.num_heads,
            args.tran_dropout,
            args.feat_dropout,
            args.prop_dropout,
            args.norm,
            args.lr,
            args.weight_decay,
            args.local_step,
        )
        for i in range(args.n_trainer)
    ]

    server = Server(
        class_num,
        x.shape[1],
        args.nlayer,
        args.hidden_dim,
        args.num_heads,
        args.tran_dropout,
        args.feat_dropout,
        args.prop_dropout,
        args.norm,
        trainers,
    )
    print("global_rounds", args.global_rounds)
    for i in range(args.global_rounds):
        server.train(i)

    results = [trainer.get_all_loss_accuracy.remote() for trainer in server.trainers]
    results = np.array([ray.get(result) for result in results])

    train_data_weights = [len(i) for i in in_com_train_data_indexes]
    test_data_weights = [len(i) for i in in_com_test_data_indexes]

    average_train_loss = np.average(
        [row[0] for row in results], weights=train_data_weights, axis=0
    )
    average_train_accuracy = np.average(
        [row[1] for row in results], weights=train_data_weights, axis=0
    )
    average_test_loss = np.average(
        [row[2] for row in results], weights=test_data_weights, axis=0
    )
    average_test_accuracy = np.average(
        [row[3] for row in results], weights=test_data_weights, axis=0
    )

    results = [trainer.local_test.remote() for trainer in server.trainers]
    results = np.array([ray.get(result) for result in results])

    average_final_test_loss = np.average(
        [row[0] for row in results], weights=test_data_weights, axis=0
    )
    average_final_test_accuracy = np.average(
        [row[1] for row in results], weights=test_data_weights, axis=0
    )

    print(average_final_test_loss, average_final_test_accuracy)

    average_final_test_loss_repeats.append(average_final_test_loss)
    average_final_test_accuracy_repeats.append(average_final_test_accuracy)

print(
    f"average_testing_loss {np.average(average_final_test_loss_repeats)} std {np.std(average_final_test_loss_repeats)}"
)
print(
    f"average_testing_accuracy {np.average(average_final_test_accuracy_repeats)} std {np.std(average_final_test_accuracy_repeats)}"
)

ray.shutdown()
