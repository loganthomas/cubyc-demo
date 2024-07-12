import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from cubyc import Run


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# @Run(tags=['fashion-mnist'])
# def train_loop(dataloader, model, batch_size, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * batch_size + len(X)
#             print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
#         # yield {'loss': loss}


# @Run(tags=['fashion-mnist'])
# def test_loop(dataloader, model, loss_fn):
    # model.eval()
    # size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    # test_loss, correct = 0, 0

    # with torch.no_grad():
    #     for X, y in dataloader:
    #         pred = model(X)
    #         test_loss += loss_fn(pred, y).item()
    #         correct += (pred.argmax(axis=1) == y).type(torch.float).sum().item()
    # test_loss /= num_batches
    # correct /= size
    # print(f'Test Error:\nAccurcacy: {(100 * correct):>0.1f}% Avg loss: {test_loss:>8f}\n')
    # # yield {'test_loss': test_loss, 'acc': correct}


@Run(tags=['fashion-mnist'])
def experiment_run(batch_size):

    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size)
    test_dataloader = DataLoader(test_data, batch_size)

    model = Network()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        print('-' * 72)

        # Train loop
        size = len(train_dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss, current = loss.item(), batch * batch_size + len(X)
            if batch % 100 == 0:
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

        # Test loop
        model.eval()
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in test_dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(axis=1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f'Test Error:\nAccurcacy: {(100 * correct):>0.1f}% Avg loss: {test_loss:>8f}\n')
        yield {'loss': loss, 'test_loss': test_loss, 'acc': correct}
    print('Done.')


if __name__ == '__main__':
    for batch_size in [32, 64, 128]:
        experiment_run(batch_size=batch_size)
