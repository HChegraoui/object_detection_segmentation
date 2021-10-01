from torch.nn import BCELoss, NLLLoss
from torch.optim import Adam

from segmentation.UNet import UNet
from segmentation.loader import DatasetLoader
import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Adam

EPSILON = 1e-34
# Press the green button in the gutter to run the script.
class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        y_input = torch.log(y_input + EPSILON)
        print(y_input.shape, y_target.shape)
        return cross_entropy(y_input, y_target.long().squeeze(1), weight=self.weight,
                             ignore_index=self.ignore_index)
if __name__ == '__main__':
    model = UNet(1,2,(3,4,8), use_bbox=False)
    dataset = DatasetLoader('./example_images', use_bbox=False)
    print(model)
    loss = LogNLLLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model.compile(loss, optimizer, './example_images')
    x = model.fit_dataset(dataset, n_epochs = 10, n_batch = 3, shuffle = True,
                    val_dataset= dataset, save_freq = 6, save_model = True,
                    verbose =  True)
    print(x)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

