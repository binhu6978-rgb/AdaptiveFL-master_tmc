"""SemConsFL on AdaptiveFL's native prefix submodels.

Engineering conventions (not additional claims about the paper):
* task CE + private semantic CE, both backpropagating into the backbone;
* d0 = full classifier input size, fixed orthogonal measurement head;
* zero-padded identity adapters and zero-initialized semantic classifiers;
* duplicate slots retain their updates; memory records one observation/client/round;
* medians use the midpoint of the two central values for even sample counts.
"""

import math
import random
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief


@dataclass(frozen=True)
class SemConsConfig:
    emb_dim: int = 128
    n_proto: int = 5
    lambda_mu: float = 0.9
    window: int = 5                 # communication rounds, NOT participations
    kappa: float = 10.0
    tau: float = 0.0
    beta_min: float = 0.05
    tau_beta_target: float = 0.3
    tau_beta_warm_rounds: int = 50
    n_trust: int = 3
    n_cons: int = 3                 # unspecified in paper; engineering default
    n_critical: int = 3
    direction_p: float = 2.0
    alpha0: float = 0.2
    delta0: float = 0.05
    eta_g: float = 1.0
    eps: float = 1e-12


@contextmanager
def isolated_rng():
    """Extra measurement work must not advance training/sampling RNGs."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _linear(in_dim, out_dim, bias):
    with isolated_rng():
        return nn.Linear(in_dim, out_dim, bias=bias)


def _finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("Nonfinite tensor: " + name)


def _rep_to_vec(rep):
    if rep.ndim == 4:
        return F.adaptive_avg_pool2d(rep, 1).flatten(1)
    if rep.ndim == 2:
        return rep
    raise ValueError("Unexpected representation shape: {}".format(tuple(rep.shape)))


def _overlap_slices(global_shape, local_shape):
    if len(global_shape) != len(local_shape) or any(
        local > full for local, full in zip(local_shape, global_shape)
    ):
        raise ValueError("Invalid prefix coverage: {} -> {}".format(local_shape, global_shape))
    return tuple(slice(0, int(size)) for size in local_shape)


def _median(values, dim=0):
    # Explicit even-count convention, consistent with numpy.median for Q/MAD.
    return torch.quantile(values, 0.5, dim=dim)


class FrozenMeasurementHead(nn.Module):
    def __init__(self, d0, out_dim=128, seed=1):
        super().__init__()
        if d0 < out_dim:
            raise ValueError("Measurement dimension must not exceed d0")
        self.proj = _linear(d0, out_dim, bias=False)
        generator = torch.Generator().manual_seed(int(seed) + 9137)
        q = torch.linalg.qr(torch.randn(d0, out_dim, generator=generator), mode="reduced").Q
        with torch.no_grad():
            self.proj.weight.copy_(q.T)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        # Frozen weights still propagate gradients to the trainable adapter.
        return self.proj(x)


class ClientSemanticBank:
    def __init__(self):
        self.adapter_states = {}
        self.classifier_states = {}

    def load(self, client_id, in_dim, d0, emb_dim, num_classes, device):
        key = (int(client_id), int(in_dim))
        adapter = _linear(in_dim, d0, bias=False)
        classifier = _linear(emb_dim, num_classes, bias=True)
        with torch.no_grad():
            adapter.weight.zero_()
            n = min(in_dim, d0)
            adapter.weight[:n, :n].copy_(torch.eye(n))
            classifier.weight.zero_()
            classifier.bias.zero_()
        if key in self.adapter_states:
            adapter.load_state_dict(self.adapter_states[key])
        if int(client_id) in self.classifier_states:
            classifier.load_state_dict(self.classifier_states[int(client_id)])
        return adapter.to(device), classifier.to(device)

    def save(self, client_id, in_dim, adapter, classifier):
        self.adapter_states[(int(client_id), int(in_dim))] = {
            k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()
        }
        self.classifier_states[int(client_id)] = {
            k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()
        }


class LocalUpdateSemConsFL:
    def __init__(self, args, dataset, idxs, client_id):
        self.args, self.dataset = args, dataset
        self.idxs, self.client_id = list(idxs), int(client_id)
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, self.idxs, args), batch_size=args.local_bs,
            shuffle=True, drop_last=True,
        )

    def _make_optimizer(self, round_idx, params):
        args = self.args
        if args.optimizer == "sgd":
            return torch.optim.SGD(params, lr=args.lr * args.lr_decay ** round_idx,
                                   momentum=args.momentum, weight_decay=args.weight_decay)
        if args.optimizer == "adam":
            return torch.optim.Adam(params, lr=args.lr)
        if args.optimizer == "adaBelief":
            return AdaBelief(params, lr=args.lr)
        raise ValueError("Unsupported optimizer: " + args.optimizer)

    def train(self, round_idx, net, bank, measurement_head, d0, cfg):
        args = self.args
        in_dim = net.classifier.in_features
        adapter, classifier = bank.load(self.client_id, in_dim, d0, cfg.emb_dim,
                                        args.num_classes, args.device)
        net.train()
        adapter.train()
        classifier.train()
        measurement_head.eval()
        optimizer = self._make_optimizer(
            round_idx, list(net.parameters()) + list(adapter.parameters()) + list(classifier.parameters())
        )
        losses = torch.zeros(2, device=args.device)
        steps = 0
        for _ in range(args.local_ep):
            for images, labels in self.ldr_train:
                images, labels = images.to(args.device), labels.to(args.device).long()
                optimizer.zero_grad(set_to_none=True)
                out = net(images)
                rep = _rep_to_vec(out["representation"])
                if rep.shape[1] != in_dim:
                    raise ValueError("Representation/classifier dimensions disagree")
                z = F.normalize(measurement_head(adapter(rep)), dim=1, eps=cfg.eps)
                task_loss = F.cross_entropy(out["output"], labels)
                semantic_loss = F.cross_entropy(classifier(z), labels)
                # Explicit engineering interpretation: semantic supervision also
                # updates the backbone. No prototype/anchor/KD term is used.
                loss = task_loss + semantic_loss
                _finite(loss, "local loss")
                loss.backward()
                optimizer.step()
                losses += torch.stack((task_loss.detach(), semantic_loss.detach()))
                steps += 1

        for name, value in net.state_dict().items():
            if torch.is_floating_point(value):
                _finite(value, name)
        for module in (adapter, classifier):
            for value in module.parameters():
                _finite(value, "private parameter")
        bank.save(self.client_id, in_dim, adapter, classifier)
        protos, counts = self._prototypes(round_idx, net, adapter, measurement_head, cfg)
        self.stats = dict(zip(("task_loss", "semantic_loss"),
                             (losses / max(steps, 1)).cpu().tolist()))
        self.stats["steps"] = steps
        # Native CPU tensors keep all returned slots off the GPU.
        state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        return state, protos, counts

    @torch.no_grad()
    def _prototypes(self, round_idx, net, adapter, head, cfg):
        net.eval()
        adapter.eval()
        head.eval()
        # Also isolate RNG used inside transforms; loader.generator alone is not enough.
        with isolated_rng():
            generator = torch.Generator().manual_seed(
                int(self.args.seed) * 1000003 + self.client_id * 97 + round_idx
            )
            loader = DataLoader(
                DatasetSplit(self.dataset, list(dict.fromkeys(self.idxs)), self.args),
                batch_size=self.args.local_bs, shuffle=False, drop_last=False, generator=generator,
            )
            sums = torch.zeros(self.args.num_classes, cfg.emb_dim, device=self.args.device)
            counts = torch.zeros(self.args.num_classes, dtype=torch.long, device=self.args.device)
            for images, labels in loader:
                labels = labels.to(self.args.device).long()
                rep = _rep_to_vec(net(images.to(self.args.device))["representation"])
                z = F.normalize(head(adapter(rep)), dim=1, eps=cfg.eps)
                _finite(z, "prototype embeddings")
                sums.index_add_(0, labels, z)
                counts += torch.bincount(labels, minlength=self.args.num_classes)
            sums, counts = sums.cpu(), counts.cpu()
        protos, valid_counts = {}, {}
        for c, n in enumerate(counts.tolist()):
            if n < cfg.n_proto:
                continue
            p = sums[c] / n
            if p.norm().item() > cfg.eps:
                protos[c] = F.normalize(p, dim=0, eps=cfg.eps)
                valid_counts[c] = n
        return protos, valid_counts


def compute_local_delta(local_state, dispatched_state):
    if local_state.keys() != dispatched_state.keys():
        raise ValueError("Local and dispatched state keys differ")
    delta = OrderedDict()
    for name, value in local_state.items():
        if value.shape != dispatched_state[name].shape:
            raise ValueError("Local shape changed: " + name)
        if torch.is_floating_point(value):
            delta[name] = value.detach().cpu().float() - dispatched_state[name].detach().cpu().float()
            _finite(delta[name], "delta " + name)
    return delta


def covered_median(deltas, full_shape):
    """Exact coordinate median without zero padding or N full-model buffers.

    Prefix shape boundaries divide the tensor into boxes with constant coverage.
    Median is computed only from actual covering tensors in each box.
    """
    from itertools import product

    reference = deltas[0].new_zeros(full_shape)
    valid = torch.zeros(full_shape, dtype=torch.bool, device=reference.device)
    for delta in deltas:
        _overlap_slices(full_shape, delta.shape)
    boundaries = [sorted({0, int(size)} | {int(d.shape[axis]) for d in deltas})
                  for axis, size in enumerate(full_shape)]
    intervals = [list(zip(b[:-1], b[1:])) for b in boundaries]
    for box in product(*intervals):
        slices = tuple(slice(lo, hi) for lo, hi in box)
        covering = [d[slices] for d in deltas
                    if all(d.shape[axis] >= hi for axis, (_, hi) in enumerate(box))]
        if covering:
            reference[slices] = _median(torch.stack(covering), dim=0)
            valid[slices] = True
    return reference, valid


class SemConsServer:
    def __init__(self, args, model_list, cfg=None):
        self.cfg = cfg or SemConsConfig()
        self.device = args.device
        self.d0 = model_list[-1].classifier.in_features
        self.head = FrozenMeasurementHead(self.d0, self.cfg.emb_dim, args.seed).to(args.device)
        # Small memory arrays stay on CPU, independent of model device.
        self.anchors = torch.zeros(args.num_classes, self.cfg.emb_dim)
        self.anchor_initialized = torch.zeros(args.num_classes, dtype=torch.bool)
        self.evidence_history = {cid: [] for cid in range(args.num_users)}
        self.critical_layers = self._select_critical_layers(model_list)
        self.last_stats = {}

    def _select_critical_layers(self, model_list):
        states = [m.state_dict() for m in model_list]
        candidates = []
        for order, (name, tensor) in enumerate(states[-1].items()):
            if tensor.ndim != 4 or not name.endswith("weight"):
                continue
            widths = [s[name].shape[0] for s in states if name in s]
            ratio = max(widths) / min(widths)
            if ratio > 1:
                candidates.append((ratio, tensor.numel(), order, name))
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return [c[-1] for c in candidates[:self.cfg.n_critical]]

    def _merge_semantic_observations(self, client_ids, protos_list, counts_list):
        buckets = {}
        for cid, protos, counts in zip(client_ids, protos_list, counts_list):
            classes = buckets.setdefault(int(cid), {})
            for c, p in protos.items():
                _finite(p, "uploaded prototype")
                classes.setdefault(c, []).append((p.cpu().float(), counts[c]))
        protos, counts = {}, {}
        for cid, classes in buckets.items():
            protos[cid], counts[cid] = {}, {}
            for c, views in classes.items():
                p = sum(p * n for p, n in views) / sum(n for _, n in views)
                if p.norm().item() > self.cfg.eps:
                    protos[cid][c] = F.normalize(p, dim=0, eps=self.cfg.eps)
                    counts[cid][c] = max(n for _, n in views)
        return protos, counts

    def _append_and_compute_reliability(self, cid, round_idx, evidence):
        history = [(r, e) for r, e in self.evidence_history[cid]
                   if round_idx - self.cfg.window + 1 <= r < round_idx]
        history.append((round_idx, evidence))
        self.evidence_history[cid] = history
        values = [e for _, e in history]
        if len(values) < 2:
            return 0.0
        return float(np.clip(np.mean(values) - math.sqrt(np.var(values, ddof=1) / len(values)), 0, 1))

    def _tau_beta(self, round_idx):
        return max(self.cfg.beta_min + 1e-6, self.cfg.tau_beta_target *
                   min(1.0, max(0.0, round_idx / self.cfg.tau_beta_warm_rounds)))

    def _memory_pass(self, round_idx, client_ids, protos_list, counts_list):
        protos, counts = self._merge_semantic_observations(client_ids, protos_list, counts_list)
        selected = sorted(set(map(int, client_ids)))
        # Expire by wall-clock communication round, even for absent clients.
        for cid, history in self.evidence_history.items():
            self.evidence_history[cid] = [(r, e) for r, e in history
                                          if r >= round_idx - self.cfg.window + 1]
        beta, observations = {}, []
        for cid in selected:
            comparable = [c for c in protos[cid] if self.anchor_initialized[c]]
            evidence, reliability = None, 0.0
            if comparable:
                total = sum(counts[cid][c] for c in comparable)
                evidence = sum(counts[cid][c] * float(torch.dot(protos[cid][c], self.anchors[c]))
                               for c in comparable) / total
                evidence = float(np.clip(evidence, -1, 1))
                reliability = self._append_and_compute_reliability(cid, round_idx, evidence)
                score = reliability / (1 + math.exp(-self.cfg.kappa * (evidence - self.cfg.tau)))
            else:
                score = 0.0
            beta[cid] = max(self.cfg.beta_min, score)
            observations.append({"client_id": cid, "evidence": evidence, "reliability": reliability,
                                 "history_count": len(self.evidence_history[cid]), "beta": beta[cid]})
        trusted = {cid for cid in selected if beta[cid] >= self._tau_beta(round_idx)}
        qualified_count = len(trusted)
        fallback = qualified_count < self.cfg.n_trust
        if fallback:
            trusted = set(selected)
        # Only now may current prototypes alter anchors used in future rounds.
        for c in range(len(self.anchors)):
            values = [protos[cid][c] for cid in selected if c in protos[cid]]
            if not values:
                continue
            median = _median(torch.stack(values))
            if median.norm().item() <= self.cfg.eps:
                continue
            current = F.normalize(median, dim=0, eps=self.cfg.eps)
            if self.anchor_initialized[c]:
                current = self.cfg.lambda_mu * self.anchors[c] + (1 - self.cfg.lambda_mu) * current
            if current.norm().item() > self.cfg.eps:
                self.anchors[c] = F.normalize(current, dim=0, eps=self.cfg.eps)
                self.anchor_initialized[c] = True
        self.last_stats = {"observations": observations, "qualified_clients": qualified_count,
                           "trusted_fallback": fallback, "reference_clients": len(trusted),
                           "initialized_anchors": int(self.anchor_initialized.sum())}
        return beta, trusted

    def _reference_slots(self, client_ids, deltas, trusted, name):
        best = {}
        for slot, cid in enumerate(client_ids):
            if cid in trusted and name in deltas[slot]:
                if cid not in best or deltas[slot][name].numel() > deltas[best[cid]][name].numel():
                    best[cid] = slot
        return list(best.values())

    def _directional_gates(self, client_ids, deltas, beta, trusted, global_state):
        factors = [[] for _ in deltas]
        layer_stats = []
        for name in self.critical_layers:
            slots = self._reference_slots(client_ids, deltas, trusted, name)
            if len(slots) < self.cfg.n_cons:
                continue
            ref, coverage = covered_median(
                [deltas[s][name].to(self.device) for s in slots], global_state[name].shape
            )
            cosines = {}
            for slot, delta_dict in enumerate(deltas):
                if name not in delta_dict:
                    continue
                delta = delta_dict[name].to(self.device)
                slices = _overlap_slices(ref.shape, delta.shape)
                mask = coverage[slices]
                a, b = delta[mask], ref[slices][mask]
                if not a.numel() or min(a.norm().item(), b.norm().item()) <= self.cfg.eps:
                    continue
                cosine = torch.dot(a, b) / (a.norm() * b.norm())
                cosines[slot] = float(cosine.clamp(-1, 1))
            if not cosines:
                continue
            # Slot-level quality retains the original dispatch multiplicity;
            # only contributors to the reference itself are deduplicated.
            values = np.asarray(list(cosines.values()))
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            quality = float(np.clip((median - mad + 1) / 2, 0, 1))
            for slot, cosine in cosines.items():
                factors[slot].append(max(0.0, cosine) ** self.cfg.direction_p *
                                     (self.cfg.alpha0 + (1 - self.cfg.alpha0) * quality))
            layer_stats.append({"name": name, "reference_clients": len(slots),
                                "cosine_median": float(median), "mad": float(mad), "quality": quality})
        self.last_stats["layers"] = layer_stats
        return [beta[int(cid)] * (sum(f) / len(f) if f else 1.0)
                for cid, f in zip(client_ids, factors)]

    def _apply_updates(self, global_state, deltas, weights, gammas):
        gamma_bar = sum(w * g for w, g in zip(weights, gammas))
        protection = 0 < gamma_bar < self.cfg.delta0
        scale = self.cfg.eta_g * (self.cfg.delta0 / gamma_bar if protection else 1.0)
        result = OrderedDict()
        for name, value in global_state.items():
            if not torch.is_floating_point(value) or gamma_bar == 0:
                result[name] = value.detach().clone()
                continue
            base = value.detach().float()
            update = torch.zeros_like(base)
            for slot, delta_dict in enumerate(deltas):
                if name in delta_dict:
                    delta = delta_dict[name].to(base.device)
                    slices = _overlap_slices(base.shape, delta.shape)
                    update[slices].add_(delta, alpha=weights[slot] * gammas[slot])
            result[name] = (base + scale * update).to(value.dtype)
            _finite(result[name], "aggregated " + name)
        self.last_stats.update(gammas=list(gammas), gamma_bar=gamma_bar,
                               protective_update=protection, applied_scale=scale)
        return result

    @torch.no_grad()
    def aggregate_round(self, round_idx, client_ids, deltas, lens, protos_list, counts_list, global_state):
        if not deltas or not (len(client_ids) == len(deltas) == len(lens) == len(protos_list) == len(counts_list)):
            raise ValueError("Empty or inconsistent slot payloads")
        if any(n <= 0 for n in lens):
            raise ValueError("Sample counts must be positive")
        beta, trusted = self._memory_pass(round_idx, client_ids, protos_list, counts_list)
        gammas = self._directional_gates(client_ids, deltas, beta, trusted, global_state)
        weights = [n / sum(lens) for n in lens]
        return self._apply_updates(global_state, deltas, weights, gammas)
