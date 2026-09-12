"""AdaptiveFL experiment loop with SemConsFL local evidence and aggregation."""

import copy
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.Fed import get_model_list, select_clients, split_model
from models.SemConsFL import (ClientSemanticBank, LocalUpdateSemConsFL, SemConsConfig,
                             SemConsServer, compute_local_delta)
from models.test import test
from utils.HeteroClients import HeteroClients
from utils.utils import get_final_acc


def SemConsFL(args, dataset_train, dataset_test, dict_users):
    if args.model != "resnet" or args.dataset != "cifar10":
        raise ValueError("First-stage SemConsFL supports the existing CIFAR-10 ResNet only")
    # Preserve independent round-0 initialization and construction RNG ordering.
    models, slim_info = get_model_list(args)
    clients = HeteroClients(args, slim_info)
    cfg = SemConsConfig()
    bank = ClientSemanticBank()
    server = SemConsServer(args, models, cfg)
    acc_list = [[] for _ in models]
    time_list, total_time = [], 0.0
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    partition = {str(int(cid)): [int(i) for i in idxs] for cid, idxs in sorted(dict_users.items())}
    encoded = json.dumps(partition, sort_keys=True, separators=(",", ":")).encode()
    metadata = {"args": {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(args).items()},
                "semcons": asdict(cfg), "models": slim_info, "critical_layers": server.critical_layers,
                "partition_sha256": hashlib.sha256(encoded).hexdigest(),
                "torch_version": torch.__version__,
                "initialization": "Independent AdaptiveFL submodels; deltas relative to exact dispatch",
                "local_objective": "task CE + semantic CE, both update backbone",
                "duplicate_policy": "all update slots; one memory observation; largest-coverage reference slot"}
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "partition.json").write_bytes(encoded)
    print("SemConsFL critical layers:", server.critical_layers)
    print("SemConsFL partition SHA256:", metadata["partition_sha256"])

    with (run_dir / "rounds.jsonl").open("w", encoding="utf-8") as trace:
        for round_idx in tqdm(range(args.epochs)):
            print("*" * 80)
            print("Round {:3d}".format(round_idx))
            m = max(int(args.frac * args.num_users), 1)
            # Same calls, ordering and duplicate behavior as Training_AdaptiveFL.
            if args.client_chosen_mode == "RL":
                ration_users = np.random.choice(range(len(models)), m)
                idx_users = clients.select_clients(ration_users)
            elif args.client_chosen_mode == "greedy":
                ration_users = np.random.choice([len(models) - 1], m)
                idx_users = random.sample(range(args.num_users), len(ration_users))
            else:
                ration_users = np.random.choice(range(len(models)), m)
                idx_users = select_clients(args, ration_users, len(models))

            deltas, lens, protos_all, counts_all, feedback, local_stats = [], [], [], [], [], []
            max_time = 0.0
            for slot, cid in enumerate(idx_users):
                begin = time.time()
                local = LocalUpdateSemConsFL(args, dataset_train, dict_users[cid], cid)
                if args.client_chosen_mode in ("RL", "random", "greedy"):
                    model_idx = clients.train(cid, ration_users[slot])
                elif args.client_chosen_mode in ("available", "fit"):
                    model_idx = ration_users[slot]
                else:
                    raise ValueError("Unknown client selection mode")
                model_idx = int(model_idx)
                feedback.append(model_idx)
                net = copy.deepcopy(models[model_idx]).to(args.device)
                dispatched = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                state, protos, counts = local.train(round_idx, net, bank, server.head, server.d0, cfg)
                deltas.append(compute_local_delta(state, dispatched))
                lens.append(len(dict_users[cid]))
                protos_all.append(protos)
                counts_all.append(counts)
                local_stats.append({"client_id": int(cid), "model_idx": model_idx, **local.stats})
                max_time = max(max_time, time.time() - begin)
                del net, state, dispatched, local

            total_time += max_time
            time_list.append(total_time)
            print("this epoch cost time:{}".format(max_time))
            print("this epoch choose: {}".format(idx_users))
            print("this epoch dispatch models: {}".format(ration_users))
            print("this epoch received models: {}".format(feedback))
            print("hetero_proportion: \t{}".format(args.client_hetero_ration))
            global_state = models[-1].state_dict()
            updated = server.aggregate_round(round_idx, idx_users, deltas, lens,
                                               protos_all, counts_all, global_state)
            for model_idx, net in enumerate(models):
                net.load_state_dict(split_model(updated, net.state_dict()))
                print(slim_info[model_idx])
                acc_list[model_idx].append(test(net, dataset_test, args))

            stats = {"round": round_idx, "client_ids": list(map(int, idx_users)),
                     "dispatch": ration_users.tolist(), "feedback": feedback, "sample_counts": lens,
                     "local": local_stats, **server.last_stats,
                     "accuracies": [acc[-1] for acc in acc_list], "full_accuracy": acc_list[-1][-1]}
            trace.write(json.dumps(stats, allow_nan=False) + "\n")
            trace.flush()
            print("SemConsFL Full={:.2f}, gamma_bar={:.6f}, qualified={}, protection={}".format(
                stats["full_accuracy"], stats["gamma_bar"], stats["qualified_clients"], stats["protective_update"]))

    # Preserve the legacy text layout/reporter, but isolate runs: the old writer
    # appends every same-day experiment to one file and can mix different seeds.
    (run_dir / "test_time.txt").write_text(
        "base " + " ".join(map(str, time_list)) + "\n", encoding="utf-8")
    file = run_dir / "accuracy.txt"
    file.write_text("".join(str(info) + " " + " ".join(map(str, acc)) + "\n"
                            for info, acc in zip(slim_info, acc_list)), encoding="utf-8")
    # Legacy reporter slices away the first two rounds; it is undefined for <=2.
    if args.epochs > 2:
        get_final_acc(file)
    if args.epochs:
        summary = {"rounds": args.epochs, "final_full": acc_list[-1][-1],
                   "best_full": max(acc_list[-1]), "accuracies": acc_list}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        torch.save({k: v.detach().cpu() for k, v in models[-1].state_dict().items()}, run_dir / "full_final.pt")
        print("SemConsFL final Full={:.2f}; best Full={:.2f}".format(summary["final_full"], summary["best_full"]))
    return acc_list
