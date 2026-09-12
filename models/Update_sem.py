#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Update_sem.py 修复版
# 修复列表：
#   [B2] 删去多余的no_grad前向（原版第一次计算结果直接丢弃）
#   [B3] proto计算改为eval模式（原版用train模式，BN/Dropout引入随机性）
#   接口兼容：train_with_proto保持返回5个值，不破坏Training_AdaptiveFL.py

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from optimizer.Adabelief import AdaBelief


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, args):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.model = args.model

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if "transformer" in self.model:
            image, label, mask = self.dataset[self.idxs[item]].values()
            return image, label, mask
        else:
            image, label = self.dataset[self.idxs[item]]
            return image, label


class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.kld_loss = nn.KLDivLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.T = 3
        self.gamma = 0

    def loss_fn_kd(self, pred, target, soft_target, gamma_active=True):
        _ce = self.ce_loss(pred, target)
        T = self.T
        if self.gamma and gamma_active:
            _kld = (self.kld_loss(self.log_softmax(pred / T), self.softmax(soft_target / T))
                    * self.gamma * T * T)
        else:
            _kld = 0
        return _ce + _kld


def _rep_to_vec(rep: torch.Tensor) -> torch.Tensor:
    """兼容[B,D]和[B,C,H,W]"""
    if rep.dim() == 4:
        return F.adaptive_avg_pool2d(rep, 1).flatten(1)
    elif rep.dim() == 2:
        return rep
    else:
        raise ValueError(f"Unexpected representation shape: {tuple(rep.shape)}")


class LocalUpdate_AdaptiveFL(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False,
                 client_id=None, num_classes=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, args),
            batch_size=self.args.local_bs,
            shuffle=True,
            drop_last=True
        )
        self.verbose = verbose
        self.idxs = idxs
        self.client_id = client_id
        self.num_classes = (int(num_classes) if num_classes is not None
                           else int(getattr(args, "num_classes", 10)))

    # ---- 原始AdaptiveFL本地训练，完全保留 ----
    def train(self, round, net):
        net.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=self.args.lr * (self.args.lr_decay ** round),
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

        Predict_loss = 0.0
        for _ in range(self.args.local_ep):
            for _, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()
                net.zero_grad()
                log_probs = net(images)["output"]
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                Predict_loss += float(loss.item())

        if self.verbose:
            print('\nUser predict Loss={:.4f}'.format(
                Predict_loss / (self.args.local_ep * len(self.ldr_train))))

        return net.state_dict()

    # ---- SemanticFL本地训练 ----
    def train_with_proto(self, round, net, semantic_server):
        """
        返回: (w, n_samples, avg_loss, protos, counts)
        保持5个返回值，与Training_AdaptiveFL.py兼容
        """
        if semantic_server is None:
            raise ValueError("semantic_server must be provided.")
        if self.client_id is None:
            raise ValueError("client_id must be provided.")

        net.train()

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                list(net.parameters()),
                lr=self.args.lr * (self.args.lr_decay ** round),
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(list(net.parameters()), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(list(net.parameters()), lr=self.args.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

        hp = getattr(semantic_server, "hp", None)
        lambda_sem = float(getattr(hp, "lambda_sem", 0.0)) if hp is not None else 0.0
        warm_T = max(1, int(getattr(hp, "sem_warmup_T", 20))) if hp is not None else 20
        t = int(getattr(semantic_server, "t", 1))
        warm_w = min(1.0, max(0.0, (t - 1) / float(warm_T)))
        lambda_sem_eff = lambda_sem * warm_w

        adapter = None
        sem_clf = None
        epoch_losses = []

        # ========================
        # 本地训练循环
        # ========================
        for _ in range(self.args.local_ep):
            batch_losses = []
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()

                out = net(images)
                logits = out["output"]
                rep = _rep_to_vec(out["representation"])

                # 懒初始化adapter和sem_clf
                if adapter is None:
                    adapter = semantic_server.get_adapter(
                        self.client_id, rep.size(1)
                    ).to(self.args.device)
                    optimizer.add_param_group({"params": adapter.parameters()})
                    if lambda_sem_eff > 0:
                        sem_clf = semantic_server.get_sem_clf(
                            self.client_id, self.num_classes
                        ).to(self.args.device)
                        optimizer.add_param_group({"params": sem_clf.parameters()})

                loss_task = self.loss_func(logits, labels)

                # [B2修复] 删去多余的no_grad前向，只计算一次
                # head.requires_grad=False保证head参数不更新，adapter有梯度正常回传
                semantic_server.head.eval()
                z = semantic_server.head(adapter(rep))  # 只计算这一次
                z = F.normalize(z, p=2, dim=1)

                if sem_clf is not None and lambda_sem_eff > 0:
                    logits_sem = sem_clf(z)
                    loss_sem = self.loss_func(logits_sem, labels)
                    loss = loss_task + lambda_sem_eff * loss_sem
                else:
                    loss = loss_task

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            if batch_losses:
                epoch_losses.append(sum(batch_losses) / len(batch_losses))

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # adapter未初始化（空数据集等极端情况）
        if adapter is None:
            return net.state_dict(), len(self.idxs), avg_loss, {}, {}

        # ========================
        # [B3修复] 计算prototypes时用eval模式
        # 避免BN/Dropout在train模式下的随机性影响prototype稳定性
        # ========================
        net.eval()
        adapter.eval()
        semantic_server.head.eval()

        sums = {}
        counts = {}

        with torch.no_grad():
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                if self.args.dataset == 'widar':
                    labels = labels.long()

                out = net(images)
                rep = _rep_to_vec(out["representation"])
                z = semantic_server.head(adapter(rep))
                z = F.normalize(z, p=2, dim=1)

                for i in range(z.size(0)):
                    c = int(labels[i].item())
                    if c not in sums:
                        sums[c] = torch.zeros(z.size(1), device=z.device)
                        counts[c] = 0
                    sums[c] += z[i].detach()
                    counts[c] += 1

        protos = {c: (sums[c] / max(1, counts[c])).cpu() for c in sums}
        counts_out = {c: int(counts[c]) for c in counts}

        # 恢复train模式，供下一轮使用
        net.train()

        return net.state_dict(), len(self.idxs), avg_loss, protos, counts_out
