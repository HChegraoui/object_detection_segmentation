import torch
from torch.autograd import Variable


def classwise_metrics(output_, gt):
    epsilon = 1e-20
    n_classes = output_.shape[1]
    if (n_classes == 1):
        t = Variable(torch.Tensor([0.5]))  # threshold
        output_ = (output_ > t).float() * 1
        start = 1
    else:
        output_, gt = torch.argmax(output_, dim=1), torch.argmax(gt, dim=1)
        start = 0
    true_positives = torch.tensor([((output_ == i) * (gt == i)).sum() for i in range(start, n_classes + start)]).float()
    selected = torch.tensor([(output_ == i).sum() for i in range(start, n_classes + start)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(start, n_classes + start)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    dice = 2 * (true_positives) / (selected + relevant + epsilon)

    return {'precision': precision, 'recall': recall, 'dice': dice}
