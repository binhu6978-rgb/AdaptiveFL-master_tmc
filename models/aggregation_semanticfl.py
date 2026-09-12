# aggregation_semanticfl.py
# SemConsFL完整修复版
# 修复列表：
#   [B1] unnormalized聚合：不再除以gate_bar（原版等价于归一化，与论文相反）
#   [B7] 非关键层gate_bar=beta_bar，eps=0.1可能放大更新，改为protective update
#   [B5] HetScore改为论文Appendix A.1的max_ch/min_ch比值
#   [B6] trusted<2时fallback到全部客户端而非返回无意义的全1过滤
#   论文对齐：beta计算改为sigma(e)*r（论文Eq.8），protective update对齐Eq.18

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# =========================
# Hyperparameters
# =========================
@dataclass
class SAFAHyperParams:
    # embedding space
    d0: int = 512
    emb_dim: int = 128
    head_hidden: int = 256

    # prototypes
    proto_min: int = 10
    proto_momentum: float = 0.9
    proto_agg: str = "median"  # "median" | "mean"

    # semantic gating (beta) -- 论文Eq.8
    # beta_{i,t} = sigma(e_{i,t}) * r_{i,t}，sigma(x) = Clip[(x+1)/2]
    warmup_rounds: int = 10
    beta_floor: float = 0.05  # [修改] 原0.2过高，降低以允许低质量客户端被充分压制

    # reliability (W-LCB)
    rel_window: int = 5          # [修改] 原20太长，论文默认W=5
    lcb_lambda: float = 1.0
    lcb_eps: float = 1e-6

    # [B1][B7] unnormalized aggregation参数（论文Eq.17-18）
    server_lr: float = 1.0
    delta_0: float = 0.05        # protective update下界（论文delta_0，应远小于正常gamma_bar）
    eps_stab: float = 1e-8       # 仅用于防除零，不影响unnormalized性质

    # Key-layer consensus filtering
    enable_key_consensus: bool = True
    consensus_start_round: int = 20

    # key-layer selection -- [B5]修复为论文HetScore = max_ch/min_ch
    key_topk: int = 3
    key_min_numel: int = 2048
    key_only_weights: bool = True

    # trusted subset
    enable_trusted_subset: bool = True
    tau_beta_min: float = 0.25
    tau_beta_max: float = 0.60
    warm_T: int = 100

    # consensus quality (论文Eq.12-13，MAD-based，参数free)
    trim_ratio: float = 0.1
    consensus_xi: float = 1e-4

    # filter系数（论文Eq.14）
    alpha_base: float = 0.5
    filter_floor: float = 0.1    # [修改] 原0.2，降低以允许高冲突层被充分压制


# =========================
# Frozen shared head（论文Section 3.2，Xavier初始化，全程冻结）
# =========================
class FrozenProjectionHead(nn.Module):
    def __init__(self, d0: int, head_hidden: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d0, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, emb_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# 工具函数
# =========================
def _l2n(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def _as_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(x, device=device, dtype=torch.float32)


def _overlap_slices(g_shape: torch.Size, c_shape: torch.Size):
    return tuple(slice(0, min(int(gs), int(cs))) for gs, cs in zip(g_shape, c_shape))


def _robust_proto_aggregate(
    vecs: List[torch.Tensor],
    wts: Optional[List[float]],
    mode: str
) -> torch.Tensor:
    if len(vecs) == 1:
        return _l2n(vecs[0])
    mode = str(mode).lower()
    if mode == "median":
        X = torch.stack(vecs, dim=0)
        return _l2n(X.median(dim=0).values)
    if wts is None or len(wts) != len(vecs):
        return _l2n(torch.stack(vecs, dim=0).mean(dim=0))
    w = torch.tensor(wts, device=vecs[0].device, dtype=torch.float32)
    w = w / (w.sum() + 1e-12)
    X = torch.stack(vecs, dim=0)
    return _l2n((X * w.view(-1, 1)).sum(dim=0))


def _cosine_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    na = float(torch.linalg.norm(a).item())
    nb = float(torch.linalg.norm(b).item())
    if na < eps or nb < eps:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb + eps))


def _trimmed_median(xs: List[float], trim_ratio: float) -> float:
    """用median替代trimmed mean，对异常值更鲁棒"""
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    n = len(xs2)
    k = int(math.floor(trim_ratio * n))
    if 2 * k >= n:
        return float(xs2[n // 2])
    core = xs2[k: n - k]
    return float(np.median(core))


def _mad(xs: List[float]) -> float:
    if not xs:
        return 0.0
    med = float(np.median(xs))
    return float(1.4826 * np.median([abs(x - med) for x in xs]))


# =========================
# Server
# =========================
class SemanticFLServer:

    def __init__(
        self,
        cfg,
        hparams: Optional[SAFAHyperParams] = None,
        device: Optional[torch.device] = None
    ):
        self.cfg = cfg
        self.device = device if device is not None else getattr(cfg, "device", torch.device("cpu"))

        hp = hparams if hparams is not None else SAFAHyperParams()
        # 允许cfg字段覆盖hp（只覆盖已有字段）
        for k in hp.__dataclass_fields__.keys():
            if hasattr(cfg, k):
                setattr(hp, k, getattr(cfg, k))
        self.hp = hp

        self.emb_dim = int(hp.emb_dim)
        self.m_min = int(hp.proto_min)
        self.t = 0

        # 固定投影头，Xavier初始化，全程冻结
        self.head = FrozenProjectionHead(hp.d0, hp.head_hidden, hp.emb_dim).to(self.device)
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.head.eval()

        self._adapter_bank: Dict[Tuple[int, int], nn.Module] = {}
        self._semclf_bank: Dict[int, nn.Module] = {}

        self.num_classes = int(getattr(cfg, "num_classes", 10))
        self.num_clients = int(getattr(cfg, "num_clients", getattr(cfg, "num_users", 0)))
        if self.num_clients <= 0:
            raise ValueError("cfg must provide num_users/num_clients > 0")

        # Reliability Memory
        self.rel_window = int(hp.rel_window)
        self.lcb_lambda = float(hp.lcb_lambda)
        self.lcb_eps = float(hp.lcb_eps)
        self.reliability: Dict[int, float] = {cid: 1.0 for cid in range(self.num_clients)}
        self._evidence_hist: Dict[int, deque] = {
            cid: deque(maxlen=self.rel_window) for cid in range(self.num_clients)
        }

        # Prototype Memory
        self.global_protos: Dict[int, torch.Tensor] = {
            c: _l2n(torch.randn(self.emb_dim, device=self.device, dtype=torch.float32))
            for c in range(self.num_classes)
        }

        self._key_layers: Optional[List[str]] = None

    # ---- 客户端helper ----
    def get_adapter(self, client_id: int, in_dim: int) -> nn.Module:
        key = (int(client_id), int(in_dim))
        if key not in self._adapter_bank:
            layer = nn.Linear(int(in_dim), int(self.hp.d0)).to(self.device)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            self._adapter_bank[key] = layer
        return self._adapter_bank[key]

    def get_sem_clf(self, client_id: int, num_classes: int) -> nn.Module:
        if client_id not in self._semclf_bank:
            clf = nn.Linear(int(self.emb_dim), int(num_classes)).to(self.device)
            nn.init.xavier_uniform_(clf.weight)
            if clf.bias is not None:
                nn.init.zeros_(clf.bias)
            self._semclf_bank[client_id] = clf
        return self._semclf_bank[client_id]

    # ---- W-LCB可靠性（论文Eq.7）----
    def _compute_wlcb(self, hist: deque) -> float:
        W_eff = len(hist)
        if W_eff == 0:
            return 1.0
        mean_e = sum(hist) / W_eff
        if W_eff < 2:
            var_e = 0.0
        else:
            var_e = sum((x - mean_e) ** 2 for x in hist) / (W_eff - 1)
        lcb = mean_e - self.lcb_lambda * math.sqrt(var_e / (W_eff + self.lcb_eps))
        return float(max(0.0, min(1.0, lcb)))

    # ---- Trusted subset阈值schedule ----
    def _tau_beta(self) -> float:
        t_eff = max(0.0, float(self.t - int(self.hp.consensus_start_round)))
        ratio = min(1.0, t_eff / max(1.0, float(self.hp.warm_T)))
        return (float(self.hp.tau_beta_min) +
                (float(self.hp.tau_beta_max) - float(self.hp.tau_beta_min)) * ratio)

    # ---- [B5修复] 关键层选择，改用论文Appendix A.1的HetScore = max_ch/min_ch ----
    def _select_key_layers(
        self,
        global_state: Dict[str, torch.Tensor],
        w_locals: List[Dict[str, torch.Tensor]]
    ) -> List[str]:
        if self._key_layers is not None:
            return self._key_layers

        K = max(0, int(self.hp.key_topk))
        if K == 0:
            self._key_layers = []
            return self._key_layers

        scores: List[Tuple[float, str]] = []
        for name, g in global_state.items():
            if int(g.numel()) < int(self.hp.key_min_numel):
                continue
            lname = name.lower()
            if bool(self.hp.key_only_weights) and ("bias" in lname or "bn" in lname):
                continue
            if g.dim() < 2:
                continue

            # [B5修复] 论文HetScore = max(d_{i,ℓ}) / min(d_{i,ℓ})，基于output channel维度
            client_out_channels = [int(g.shape[0])]  # global作为参照
            for c_state in w_locals:
                if name in c_state:
                    client_out_channels.append(int(c_state[name].shape[0]))

            if len(client_out_channels) < 2:
                continue

            max_ch = max(client_out_channels)
            min_ch = min(client_out_channels)
            het_score = float(max_ch) / float(max(1, min_ch))

            # 只选存在真实异构的层（het_score > 1.0）
            if het_score > 1.0:
                scores.append((het_score, name))

        scores.sort(key=lambda x: x[0], reverse=True)
        self._key_layers = [n for _, n in scores[:K]]
        print(f"[SemConsFL] Key layers (HetScore): {self._key_layers}")
        return self._key_layers

    # ---- 关键层共识过滤（论文Eq.11-14）----
    def _key_layer_filters(
        self,
        layer_name: str,
        g_param: torch.Tensor,
        w_locals: List[Dict[str, torch.Tensor]],
        trusted_idx: List[int],
    ) -> Tuple[List[float], float]:
        """
        返回: (filters_per_client, Q)
        filters: List[float]，长度=len(w_locals)
        Q: 共识质量 in [0,1]
        """
        # [B6修复] trusted_idx为空时用全部客户端（而非返回无意义的全1）
        if not trusted_idx:
            trusted_idx = list(range(len(w_locals)))

        # 计算trusted客户端的delta（全局空间，overlap外补零）
        deltas_trusted: List[torch.Tensor] = []
        for idx in trusted_idx:
            if idx >= len(w_locals):
                continue
            c_state = w_locals[idx]
            if layer_name not in c_state:
                continue
            c = c_state[layer_name].to(self.device, dtype=torch.float32)
            if c.dim() != g_param.dim():
                continue
            sl = _overlap_slices(g_param.shape, c.shape)
            delta = torch.zeros_like(g_param)
            delta[sl] = c[sl] - g_param[sl]
            deltas_trusted.append(delta)

        # trusted客户端不足时fallback（[B6修复]）
        if len(deltas_trusted) < 2:
            # 尝试用全部客户端
            deltas_all: List[torch.Tensor] = []
            for c_state in w_locals:
                if layer_name not in c_state:
                    continue
                c = c_state[layer_name].to(self.device, dtype=torch.float32)
                if c.dim() != g_param.dim():
                    continue
                sl = _overlap_slices(g_param.shape, c.shape)
                delta = torch.zeros_like(g_param)
                delta[sl] = c[sl] - g_param[sl]
                deltas_all.append(delta)

            if len(deltas_all) < 2:
                # 真的没有足够信息，返回neutral（不过滤）
                return [float(self.hp.alpha_base) for _ in range(len(w_locals))], 0.0
            deltas_trusted = deltas_all

        # 共识方向：coordinate-wise median（论文Eq.11）
        X = torch.stack(deltas_trusted, dim=0)
        g_cons = X.median(dim=0).values

        if float(torch.linalg.norm(g_cons).item()) < float(self.hp.consensus_xi):
            return [float(self.hp.alpha_base) for _ in range(len(w_locals))], 0.0

        # 共识质量Q（论文Eq.12-13，parameter-free）
        cos_all: List[float] = []
        for c_state in w_locals:
            if layer_name not in c_state:
                cos_all.append(0.0)
                continue
            c = c_state[layer_name].to(self.device, dtype=torch.float32)
            if c.dim() != g_param.dim():
                cos_all.append(0.0)
                continue
            sl = _overlap_slices(g_param.shape, c.shape)
            delta = torch.zeros_like(g_param)
            delta[sl] = c[sl] - g_param[sl]
            cos_all.append(_cosine_flat(delta, g_cons))

        mu_q = _trimmed_median(cos_all, float(self.hp.trim_ratio))
        sigma_q = _mad(cos_all)
        # 论文Eq.13：Q = Clip[(mu - sigma + 1)/2]
        Q = float(np.clip((mu_q - sigma_q + 1.0) / 2.0, 0.0, 1.0))

        # 过滤系数（论文Eq.14）
        base = float(self.hp.alpha_base) + (1.0 - float(self.hp.alpha_base)) * Q
        floor = float(self.hp.filter_floor)
        filters: List[float] = []
        for cosv in cos_all:
            relu_cos = max(0.0, cosv)
            # 论文Eq.14：s = relu(cos) * [alpha_0 + (1-alpha_0)*Q]
            s = relu_cos * base
            s = max(floor, s)
            filters.append(float(s))

        return filters, Q

    # ---- 主聚合函数 ----
    @torch.no_grad()
    def aggregate(
        self,
        big_model: nn.Module,
        w_locals: List[Dict[str, torch.Tensor]],
        lens: List[int],
        client_ids: List[int],
        protos_locals: List[Dict[int, Any]],
        counts_locals: List[Dict[int, int]],
        return_stats: bool = False,
    ):
        assert len(w_locals) == len(lens) == len(client_ids), \
            f"长度不一致: w_locals={len(w_locals)}, lens={len(lens)}, client_ids={len(client_ids)}"
        assert len(client_ids) == len(protos_locals) == len(counts_locals), \
            f"长度不一致: protos/counts"

        self.t += 1
        N = len(w_locals)

        # FedAvg权重
        tot = float(sum(lens)) + 1e-12
        wts = [float(n) / tot for n in lens]

        # 使用t-1轮anchors计算evidence（论文Section 3.3）
        mu_prev = {c: v.detach().clone() for c, v in self.global_protos.items()}

        sbar_list: List[float] = []
        betas: List[float] = []

        # ========================
        # Step 1: 计算beta（论文Eq.5+7+8）
        # ========================
        for cid, protos, counts in zip(client_ids, protos_locals, counts_locals):
            cid = int(cid)
            if cid not in self._evidence_hist:
                self._evidence_hist[cid] = deque(maxlen=self.rel_window)
            if cid not in self.reliability:
                self.reliability[cid] = 1.0

            # 频率加权余弦相似度（论文Eq.5）
            num, den = 0.0, 0.0
            for c, p in (protos or {}).items():
                c = int(c)
                cnt = int((counts or {}).get(c, 0))
                if cnt < int(self.hp.proto_min):
                    continue
                if c not in mu_prev:
                    continue
                pt = _l2n(_as_tensor(p, self.device))
                sim = float(torch.dot(pt, mu_prev[c]).item())
                num += float(cnt) * sim
                den += float(cnt)

            sbar = float(num / den) if den > 0 else 0.0
            sbar_list.append(sbar)

            # 可靠性记忆（论文Eq.6-7），截断负值（保守设计）
            hist = self._evidence_hist[cid]
            hist.append(max(0.0, sbar))
            r = self._compute_wlcb(hist)
            self.reliability[cid] = r

            # beta = sigma(e_{i,t}) * r_{i,t}（论文Eq.8）
            # sigma(x) = Clip[(x+1)/2]（论文默认）
            if int(self.t) <= int(self.hp.warmup_rounds):
                beta = 1.0
            else:
                sigma_e = float(np.clip((sbar + 1.0) / 2.0, 0.0, 1.0))
                beta = sigma_e * r

            beta = max(float(self.hp.beta_floor), float(beta))
            betas.append(beta)

        # 有效聚合强度 gamma_bar（论文Eq.16）
        beta_bar = float(sum(w * b for w, b in zip(wts, betas)))

        # ========================
        # Step 2: 关键层处理
        # ========================
        global_state = big_model.state_dict()
        use_key = (bool(self.hp.enable_key_consensus) and
                   int(self.t) >= int(self.hp.consensus_start_round))

        key_layers: List[str] = []
        layer_filters: Dict[str, List[float]] = {}
        layer_Q: Dict[str, float] = {}

        if use_key:
            key_layers = self._select_key_layers(global_state, w_locals)
            if key_layers:
                if bool(self.hp.enable_trusted_subset):
                    tau_b = self._tau_beta()
                    trusted_idx = [i for i, b in enumerate(betas) if b >= tau_b]
                    # [B6修复] trusted不足时用全部
                    if len(trusted_idx) < 2:
                        trusted_idx = list(range(N))
                else:
                    trusted_idx = list(range(N))

                for name in key_layers:
                    if name not in global_state:
                        continue
                    g_param = global_state[name].to(self.device, dtype=torch.float32)
                    f, Q = self._key_layer_filters(name, g_param, w_locals, trusted_idx)
                    layer_filters[name] = f
                    layer_Q[name] = Q

        # ========================
        # Step 3: [B1+B7修复] Unnormalized聚合（论文Eq.17-18）
        # 核心：不除以gate_bar，让有效步长随冲突自然收缩
        # ========================
        new_state = OrderedDict()
        protect_count = 0  # 统计protective update触发次数

        for name, g_param in global_state.items():
            g = g_param.to(torch.float32)
            acc = torch.zeros_like(g, dtype=torch.float32)
            is_key = (name in layer_filters)

            if is_key:
                # 关键层：beta * filter双重门控
                filters = layer_filters[name]
                gamma_bar_layer = 0.0

                for i, (c_state, w, beta) in enumerate(zip(w_locals, wts, betas)):
                    gamma_i = float(beta) * float(filters[i])
                    gamma_bar_layer += float(w) * gamma_i

                    if name not in c_state:
                        continue
                    c = c_state[name].to(torch.float32)
                    if c.dim() != g.dim():
                        continue
                    sl = _overlap_slices(g.shape, c.shape)
                    # overlap-only delta（关键层不做zero-fill）
                    delta = torch.zeros_like(g)
                    delta[sl] = c[sl] - g[sl]
                    acc += delta * float(w) * gamma_i

                # [B1修复] Unnormalized聚合 + Protective update（论文Eq.17-18）
                kappa_t = float(gamma_bar_layer)
                delta_0 = float(self.hp.delta_0)

                if kappa_t < delta_0:
                    # Protective update：沿门控方向做小步更新，防止停滞
                    protect_count += 1
                    acc_norm = float(acc.norm().item())
                    if acc_norm > float(self.hp.eps_stab):
                        # 归一化方向 * delta_0（论文Eq.18）
                        out = g + float(self.hp.server_lr) * delta_0 * (
                            acc / (kappa_t + float(self.hp.eps_stab))
                        )
                    else:
                        out = g  # 完全无信号时跳过
                else:
                    # 标准unnormalized（论文Eq.17）：直接乘，不归一化
                    out = g + float(self.hp.server_lr) * acc

            else:
                # 非关键层：zero-fill + beta门控（原始设计保留）
                for c_state, w, beta in zip(w_locals, wts, betas):
                    if name not in c_state:
                        continue
                    c = c_state[name].to(torch.float32)
                    padded = torch.zeros_like(g, dtype=torch.float32)
                    if c.dim() == g.dim():
                        sl = _overlap_slices(g.shape, c.shape)
                        padded[sl] = c[sl]
                    acc += (padded - g) * float(w) * float(beta)

                # [B7修复] 非关键层也用unnormalized + protective update
                kappa_t = float(beta_bar)
                delta_0 = float(self.hp.delta_0)

                if kappa_t < delta_0:
                    protect_count += 1
                    acc_norm = float(acc.norm().item())
                    if acc_norm > float(self.hp.eps_stab):
                        out = g + float(self.hp.server_lr) * delta_0 * (
                            acc / (kappa_t + float(self.hp.eps_stab))
                        )
                    else:
                        out = g
                else:
                    # [B7修复] 直接乘，不除以beta_bar（原代码除以eps=0.1会放大10倍）
                    out = g + float(self.hp.server_lr) * acc

            new_state[name] = out.to(g_param.dtype)

        # ========================
        # Step 4: 更新Prototype Memory（论文Eq.4）
        # ========================
        for c in range(self.num_classes):
            vecs: List[torch.Tensor] = []
            wcounts: List[float] = []
            for protos, counts in zip(protos_locals, counts_locals):
                if protos is None or counts is None:
                    continue
                if c not in protos:
                    continue
                cnt = int(counts.get(c, 0))
                if cnt < int(self.hp.proto_min):
                    continue
                v = _l2n(_as_tensor(protos[c], self.device))
                vecs.append(v)
                wcounts.append(float(cnt))

            if not vecs:
                continue

            proto_hat = _robust_proto_aggregate(
                vecs,
                wts=wcounts if str(self.hp.proto_agg).lower() == "mean" else None,
                mode=str(self.hp.proto_agg),
            )
            mu_old = self.global_protos[c]
            mu_new = _l2n(
                float(self.hp.proto_momentum) * mu_old +
                (1.0 - float(self.hp.proto_momentum)) * proto_hat
            )
            self.global_protos[c] = mu_new

        if return_stats:
            stats = {
                "round": int(self.t),
                "beta_bar": float(beta_bar),
                "sbar_mean": float(np.mean(sbar_list) if sbar_list else 0.0),
                "sbar_min": float(np.min(sbar_list) if sbar_list else 0.0),
                "sbar_max": float(np.max(sbar_list) if sbar_list else 0.0),
                "use_key": bool(use_key and bool(key_layers)),
                "num_key_layers": int(len(key_layers)),
                "avg_Q": float(np.mean(list(layer_Q.values())) if layer_Q else 0.0),
                "protect_count": int(protect_count),
                # 用于复现论文Fig.5的诊断量
                "sbar_mean_as_gamma": float(np.mean(sbar_list) if sbar_list else 0.0),
            }
            return new_state, stats

        return new_state


# Export alias
SemanticFLHyperParams = SAFAHyperParams
