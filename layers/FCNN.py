import torch
from torch.nn.functional import leaky_relu


class FCNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FCNN, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, dim_out)

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        h = leaky_relu(self.fc1(x))
        h = leaky_relu(self.fc2(h))
        out = self.fc3(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        train_loss = 0

        self.train()
        for x, y in data_loader:
            preds = self(x.cuda())
            loss = criterion(y.cuda(), preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x.cuda()).cpu()
