# data.py
import os
import json
import numpy as np
from torchvision import datasets, transforms
from utils.tinyimagenet import TinyImageNet


CIFAR_ROOT = os.environ.get(
    "SEMCONSFL_CIFAR_ROOT", r"C:\Users\admin\Desktop\paper_ori\all_data\data")

# -------------------------
# Record I/O (reproducible)
# -------------------------
def _build_record_path(cfg) -> str:
    """
    IMPORTANT: include every split-affecting hyperparam in filename.
    This avoids silently reusing an old split when only seed changes.
    """
    os.makedirs("data", exist_ok=True)

    # optional knobs (safe defaults if cfg doesn't have them)
    min_size = int(getattr(cfg, "min_size", 10))
    balanced = bool(getattr(cfg, "balanced", True))
    min_classes = int(getattr(cfg, "min_classes", 0))
    max_retry = int(getattr(cfg, "max_retry", 500))

    file = os.path.join(
        "data",
        f"{cfg.dataset}_U{cfg.num_users}_seed{cfg.seed}"
    )

    if bool(cfg.iid):
        file += "_iid"
    else:
        file += f"_dirichlet_a{float(cfg.data_beta)}"
        file += "_bal" if balanced else "_unbal"
        file += f"_minsz{min_size}"
        if min_classes > 0:
            file += f"_minc{min_classes}"
        # max_retry generally doesn't need to be in the filename, but harmless:
        file += f"_try{max_retry}"

    return file + ".json"


def _save_record(path, cfg, dict_users, extra_meta=None):
    payload = {
        "dataset": cfg.dataset,
        "num_users": int(cfg.num_users),
        "iid": bool(cfg.iid),
        "data_beta": float(cfg.data_beta),
        "seed": int(cfg.seed),
        "extra_meta": extra_meta or {},
        "train_data": dict_users,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_record(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    train_data = payload["train_data"]
    # json may store keys as strings
    train_data = {int(k): v for k, v in train_data.items()}
    return train_data, payload.get("extra_meta", {})


# -------------------------
# Dataset prepare
# -------------------------
def prepare_tiny_imagenet(data_dir=r"C:\Users\admin\Desktop\paper_ori\all_data\data\tiny-imagenet-200/"):
    trans_imagenet_train = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2770, 0.2691, 0.2821]),
    ])
    trans_imagenet_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2770, 0.2691, 0.2821]),
    ])

    # 你已有的 TinyImageNet dataset 类
    dataset_train = TinyImageNet(data_dir, train=True, transform=trans_imagenet_train)
    dataset_test  = TinyImageNet(data_dir, train=False, transform=trans_imagenet_val)
    return dataset_train, dataset_test


def _cifar_transforms(mean, std, train_augmentation='none'):
    if train_augmentation not in {'none', 'paper'}:
        raise ValueError(
            'unsupported CIFAR train augmentation: {}'.format(
                train_augmentation))
    train_steps = []
    if train_augmentation == 'paper':
        train_steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose(train_steps)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform_train, transform_test


def prepare_cifar10(train_augmentation='none'):
    transform_train, transform_test = _cifar_transforms(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
        train_augmentation=train_augmentation)
    trainset = datasets.CIFAR10(root=CIFAR_ROOT, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=CIFAR_ROOT, train=False, download=True, transform=transform_test)
    return trainset, testset


def prepare_cifar100(train_augmentation='none'):
    transform_train, transform_test = _cifar_transforms(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761),
        train_augmentation=train_augmentation)
    trainset = datasets.CIFAR100(root=CIFAR_ROOT, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=CIFAR_ROOT, train=False, download=True, transform=transform_test)
    return trainset, testset


# -------------------------
# Splits
# -------------------------
def iid_split(trainset, num_clients, seed=1):
    rng = np.random.RandomState(seed)
    all_idxs = np.arange(len(trainset))
    rng.shuffle(all_idxs)
    splits = np.array_split(all_idxs, num_clients)
    return [s.astype(int).tolist() for s in splits]


def _targets_array(trainset):
    # torchvision CIFAR10/100 use .targets
    y = np.array(trainset.targets)
    return y.astype(int)


def _client_class_hist(client_indices, labels, num_classes):
    hist = np.zeros((len(client_indices), num_classes), dtype=int)
    for cid, idxs in enumerate(client_indices):
        if len(idxs) == 0:
            continue
        lbl = labels[np.array(idxs, dtype=int)]
        hist[cid] = np.bincount(lbl, minlength=num_classes)
    return hist


def dirichlet_split_unbalanced(trainset, num_clients, alpha=0.5, seed=1, min_size=10, max_retry=500, min_classes=0):
    """
    Classic Dirichlet label skew (UNBALANCED total sizes may happen).
    """
    rng = np.random.RandomState(seed)
    labels = _targets_array(trainset)
    num_classes = int(labels.max() + 1)

    for _ in range(max_retry):
        client_indices = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, cuts)

            for cid, part in enumerate(splits):
                client_indices[cid].extend(part.astype(int).tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) < min_size:
            continue

        if min_classes > 0:
            hist = _client_class_hist(client_indices, labels, num_classes)
            ncls = (hist > 0).sum(axis=1)
            if ncls.min() < min_classes:
                continue

        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])
        return client_indices

    raise RuntimeError(
        f"dirichlet_split_unbalanced failed after {max_retry} retries. "
        f"Try larger alpha, smaller min_size/min_classes, or fewer clients."
    )


def dirichlet_split_balanced(trainset, num_clients, alpha=0.5, seed=1, min_size=10, max_retry=500, min_classes=0):
    """
    Top-conference-friendly split used widely in FL papers:
    - Dirichlet label-skew
    - Balanced total samples per client (quantity-balanced)
    - Optional constraints: min_size, min_classes per client
    """
    rng = np.random.RandomState(seed)
    labels = _targets_array(trainset)
    num_classes = int(labels.max() + 1)
    n = len(labels)

    # fixed target size per client (balanced, remainder distributed)
    base = n // num_clients
    rem = n % num_clients
    target = np.full(num_clients, base, dtype=int)
    if rem > 0:
        target[:rem] += 1  # deterministic; seed affects allocation later
    assert target.sum() == n

    # pre-shuffle per-class pools once per try (we will reshuffle each try)
    class_pools = [np.where(labels == k)[0] for k in range(num_classes)]

    for _ in range(max_retry):
        client_indices = [[] for _ in range(num_clients)]
        remaining = target.copy()

        # shuffle each class pool for this attempt
        pools = []
        for k in range(num_classes):
            idx_k = class_pools[k].copy()
            rng.shuffle(idx_k)
            pools.append(idx_k)

        # assign class-by-class with capacity-aware sampling
        for k in range(num_classes):
            idx_k = pools[k]
            # base dirichlet preference for this class
            pref = rng.dirichlet(np.repeat(alpha, num_clients)).astype(float)

            for idx in idx_k:
                mask = (remaining > 0)
                if not mask.any():
                    break  # should not happen

                probs = pref * mask
                s = probs.sum()
                if s <= 0:
                    # if all masked out by numerical issues, fall back to uniform over available clients
                    probs = mask.astype(float)
                    probs /= probs.sum()
                else:
                    probs /= s

                cid = int(rng.choice(num_clients, p=probs))
                client_indices[cid].append(int(idx))
                remaining[cid] -= 1

        # sanity: all clients reach target
        if not np.all(remaining == 0):
            continue

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) < min_size:
            continue

        if min_classes > 0:
            hist = _client_class_hist(client_indices, labels, num_classes)
            ncls = (hist > 0).sum(axis=1)
            if ncls.min() < min_classes:
                continue

        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])
        return client_indices

    raise RuntimeError(
        f"dirichlet_split_balanced failed after {max_retry} retries. "
        f"Try larger alpha, smaller min_size/min_classes, or fewer clients."
    )
# -------------------------
# Main API
# -------------------------
def get_dataset(cfg):
    """
    Keep API consistent with main:
        train_set, test_set, dict_users

    Required cfg fields:
        dataset in {"cifar10","cifar100","tiny_imagenet"}
        num_users: int
        iid: bool
        data_beta: float (Dirichlet alpha)
        seed: int
        generate_data: bool

    Optional cfg fields:
        balanced: bool
        min_size: int
        min_classes: int
        max_retry: int
        data_dir: str  (for tiny-imagenet path)
    """
    if cfg.dataset not in {"cifar10", "cifar100", "tiny_imagenet"}:
        raise ValueError("Only cifar10, cifar100, tiny_imagenet are supported.")

    record_path = _build_record_path(cfg)

    train_augmentation = str(getattr(
        cfg, "cifar_train_augmentation", "none"))
    if cfg.dataset == "cifar10":
        train_set, test_set = prepare_cifar10(train_augmentation)
    elif cfg.dataset == "cifar100":
        train_set, test_set = prepare_cifar100(train_augmentation)
    else:
        data_dir = getattr(cfg, "data_dir", r"C:\Users\admin\Desktop\paper_ori\all_data\data/tiny-imagenet-200/")
        train_set, test_set = prepare_tiny_imagenet(data_dir=data_dir)

    # load cached split
    if (not bool(cfg.generate_data)) and os.path.exists(record_path):
        dict_users, _ = _load_record(record_path)
        return train_set, test_set, dict_users

    # split params
    seed = int(cfg.seed)
    num_users = int(cfg.num_users)
    alpha = float(cfg.data_beta)

    min_size = int(getattr(cfg, "min_size", 10))
    min_classes = int(getattr(cfg, "min_classes", 0))
    max_retry = int(getattr(cfg, "max_retry", 500))
    balanced = bool(getattr(cfg, "balanced", True))

    # 你原来强制 balanced=False 的行为保留与否随你
    balanced = False

    # generate fresh
    if bool(cfg.iid):
        client_indices = iid_split(train_set, num_users, seed=seed)
        extra_meta = {"split": "iid", "seed": seed}
    else:
        if balanced:
            client_indices = dirichlet_split_balanced(
                train_set, num_users, alpha=alpha, seed=seed,
                min_size=min_size, max_retry=max_retry, min_classes=min_classes
            )
            split_name = "dirichlet_balanced"
        else:
            client_indices = dirichlet_split_unbalanced(
                train_set, num_users, alpha=alpha, seed=seed,
                min_size=min_size, max_retry=max_retry, min_classes=min_classes
            )
            split_name = "dirichlet_unbalanced"

        extra_meta = {
            "split": split_name,
            "alpha": alpha,
            "seed": seed,
            "min_size": min_size,
            "min_classes": min_classes,
            "max_retry": max_retry,
        }

    dict_users = {i: client_indices[i] for i in range(num_users)}
    _save_record(record_path, cfg, dict_users, extra_meta=extra_meta)
    return train_set, test_set, dict_users
