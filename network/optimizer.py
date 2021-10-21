import torch


def train_loop(dataloader, model, loss_fn, optimizer, device):
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device)).reshape(len(y), -1, 4)
        loss_train = loss_fn(pred, y.to(device))

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if batch % 10000 == 0:
            loss = loss_train.item()
            print(f"Train loss: {loss}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_test, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            yc = y.to(device)
            pred = model(X.to(device))
            loss_test += loss_fn(pred, yc).item()
            correct += (pred.argmax(1) == yc).type(torch.float).sum().item()

    loss_test /= num_batches
    correct /= size
    print(f"Accuracy: {100 * correct}, Avg loss: {loss_test} \n")