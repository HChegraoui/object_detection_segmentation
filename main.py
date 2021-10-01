from torch.nn import BCELoss
from torch.optim import Adam

from segmentation.UNet import UNet
from segmentation.loader import DatasetLoader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = UNet(1,1,(3,4,8), use_bbox=False)
    dataset = DatasetLoader('./example_images', use_bbox=False)

    loss = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model.compile(loss, optimizer, './example_images')
    x = model.fit_dataset(dataset, n_epochs = 10, n_batch = 3, shuffle = True,
                    val_dataset= dataset, save_freq = 6, save_model = True,
                    verbose =  True)
    print(x)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

