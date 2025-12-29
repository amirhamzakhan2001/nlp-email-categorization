import torch
from sklearn.metrics import accuracy_score


def evaluate(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(
            model(torch.tensor(X_test).to(device)),
            dim=1
        ).cpu().numpy()

    return accuracy_score(y_test, preds)