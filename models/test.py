import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def test(net_glob, dataset_test, args):
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test


def test_img(net_g, datatest, args):
    net_g.eval()

    data_loader = DataLoader(
        datatest,
        batch_size=getattr(args, "test_bs", args.bs),
        shuffle=False,
        num_workers=0,   # 先用0排查，稳定后再试2/4
        pin_memory=True
    )

    use_cuda = (args.gpu != -1) and torch.cuda.is_available()

    if use_cuda:
        device = args.device
        test_loss = torch.zeros(1, device=device)
        correct = torch.zeros(1, device=device)
    else:
        device = torch.device("cpu")
        test_loss = 0.0
        correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            # target 统一转 tensor
            if not torch.is_tensor(target):
                target = torch.tensor(target)

            # 保证 target 至少是一维
            if target.dim() == 0:
                target = target.unsqueeze(0)

            target = target.long()

            if use_cuda:
                if not torch.is_tensor(data):
                    data = torch.tensor(data)
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

            output = net_g(data)
            log_probs = output['output'] if isinstance(output, dict) else output

            if use_cuda:
                test_loss += F.cross_entropy(log_probs, target, reduction='sum')
                pred = log_probs.argmax(dim=1)
                correct += (pred == target).sum()
            else:
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                pred = log_probs.argmax(dim=1)
                correct += (pred == target).sum().item()

    if use_cuda:
        test_loss = (test_loss / len(data_loader.dataset)).item()
        accuracy = (100.0 * correct / len(data_loader.dataset)).item()
        correct_num = int(correct.item())
    else:
        test_loss = test_loss / len(data_loader.dataset)
        accuracy = 100.0 * correct / len(data_loader.dataset)
        correct_num = int(correct)

    if args.verbose:
        print(
            '\nTest set: Average loss: {:.4f}\nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct_num, len(data_loader.dataset), accuracy
            )
        )

    return accuracy, test_loss