import torch
from torch.utils.data import DataLoader, TensorDataset
from numpy import max
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn import Softmax

class ClfHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClfHead, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, in_channels // 2)
        self.linear2 = torch.nn.Linear(in_channels // 2, out_channels)
        #self.linear3 = torch.nn.Linear(in_channels // 4, out_channels)
        #self.linear4 = torch.nn.Linear(in_channels // 8, out_channels)
    def forward(self, x):
        x = self.linear1(x).relu()
        #x = self.linear2(x).relu()
        return self.linear2(x)


def train_clf_head(dataset, epochs=200):
    nu_labels = max(dataset[1]) + 1
    clf_head = ClfHead(dataset[0].shape[1], nu_labels)
    print(clf_head)
    data_tensors = TensorDataset(torch.Tensor(dataset[0]), torch.Tensor(dataset[1]).type(torch.LongTensor))
    loader = DataLoader(data_tensors, batch_size=8)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(clf_head.parameters(), lr=3e-4)
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            opt.zero_grad()
            predictions = clf_head(batch[0])
            loss = loss_fn(predictions, batch[1])
            total_loss += loss.item()
            loss.backward()
            opt.step()
    print(f'EPOCH: {epoch}   LOSS: {total_loss / len(loader)}')
    total_acc = 0
    for batch in loader:
        predictions = clf_head(batch[0])
        total_acc += sum(torch.argmax(predictions, axis=1) == batch[1])
    print(f'TRAINING ACC: { total_acc / dataset[0].shape[0]}')

    return clf_head

    

def clf_head_inference(clf_head, dataset):
    data_tensors = TensorDataset(torch.Tensor(dataset[0]), torch.Tensor(dataset[1]).type(torch.LongTensor))
    loader = DataLoader(data_tensors, batch_size=8)
    total_acc = 0
    for batch in loader:
        predictions = clf_head(batch[0])
        total_acc += sum(torch.argmax(predictions, axis=1) == batch[1]).numpy()
    pred_probs = Softmax(dim=1)(clf_head(torch.Tensor(dataset[0]))).detach().numpy()
    predictions = torch.argmax(clf_head(torch.Tensor(dataset[0])), dim=1)
    print(f'ACC: { (total_acc / dataset[0].shape[0])}')
    f = f1_score(dataset[1], predictions, average='macro')
    auc = roc_auc_score(dataset[1], pred_probs, multi_class='ovr')
    print(f'F1 Score: {f}')
    print(f'AUC: {auc}')
    return {'accuracy': total_acc / dataset[0].shape[0], 'f1': f, 'auc': auc}