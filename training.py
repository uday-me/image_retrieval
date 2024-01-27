import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import EmbeddingsDataset, CLASS_NAMES, EmbeddingsTestDataset
from test import evaluate_model
from utils import MultiPosConLoss, get_mAP, MLPResidualAdapter


input_dim = 1024
hidden_dims = [512, 512, 512]
batch_size = 1024
learning_rate = 0.0001
dropout_rate = 0.25


model = MLPResidualAdapter(input_dim=input_dim, 
                           hidden_dims=hidden_dims,
                           dropout=dropout_rate)
contrastive_loss = MultiPosConLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

training_data = EmbeddingsDataset(is_train=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# load the validation set in memory:
val_data = EmbeddingsTestDataset('val')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# train:
n_epochs = 20
for epoch_idx in range(n_epochs):
    for batch_idx, (batch_labels, batch_data) in enumerate(train_dataloader):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        # forward pass:
        model.train()
        outputs = model(batch_data)

        loss = contrastive_loss(outputs.to(device), batch_labels.to(device))

        # get metrics:
        batch_map = get_mAP(outputs, batch_labels)
        print(f"epoch: {epoch_idx}, batch_idx: {batch_idx}, "
              f"loss: {loss['loss']}, mAP: {batch_map}")

        # optimize:
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        # free-up:
        del batch_data
        del batch_labels

    # Eval the model on test data:
    model.eval()
    test_mAP = evaluate_model(val_data, model, ks=[1, 10, 50, 100, 200])
    print(f'TEST mAP: ', test_mAP)
    # Checkpoint the model:
    checkpoint = {
        'epoch': epoch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    model_str = '-'.join([str(n) for n in hidden_dims])
    torch.save(checkpoint, f'checkpoint-ep-{epoch_idx}-l-{model_str}.pth')