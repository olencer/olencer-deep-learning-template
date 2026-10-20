from datasets.example import ExampleDataset
from models.example import ExampleModel

from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("./logs/")
Path("./ckpts/").mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])

train = ExampleDataset(train=True, transform=transform)
val   = ExampleDataset(train=False)

train_loader = DataLoader(train, 1024, True, num_workers=64)
val_loader   = DataLoader(val,  1024, num_workers=64)

model = ExampleModel()
model.to(device)
model_name = type(model).__name__

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)

epochs = 300
best_val_loss = float("inf")
best_val_acc = 0
bad_epochs = 0
patience = int(epochs * 0.2)

epoch_progress_bar = tqdm(range(1, epochs + 1))
for epoch in epoch_progress_bar:
    epoch_progress_bar.set_description(f"Epoch {epoch} / {epochs}")

    model.train()
    total_train_loss = 0
    total_train_num = 0
    for x, t in train_loader:
        x, t = x.to(device), t.to(device)
        y = model(x)
        loss = loss_fn(y, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * t.size(0)
        total_train_num += t.size(0)

    train_loss = total_train_loss / total_train_num
    writer.add_scalar("train/loss", train_loss, epoch)

    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val_num = 0
    with torch.no_grad():
        for x, t in val_loader:
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = loss_fn(y, t)

            total_val_loss += loss.item() * t.size(0)
            total_val_correct += (y.argmax(1) == t).sum().item()
            total_val_num += t.size(0)

    val_loss = total_val_loss    / total_val_num
    val_acc  = total_val_correct / total_val_num

    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/acc" ,  val_acc, epoch)

    if epoch % 20 == 0:
        save_path = (
            f"./ckpts/{model_name}"
            f"_epoch-{epoch}"
            f"_acc-{val_acc:.2f}"
            f"_val-{val_loss:.6f}.pth"
        )
        torch.save(model.state_dict(), save_path)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        bad_epochs = 0

        save_path = f"./ckpts/{model_name}_best-loss.pth"
        Path(save_path.replace("pth", "txt")).write_text(
            f"epoch-{epoch}_loss-{best_val_loss:.6f}",
            encoding="utf-8"
        )
        torch.save(model.state_dict(), save_path)
    else:
        bad_epochs += 1

    if val_acc >= best_val_acc:
        best_val_acc = val_acc

        save_path = f"./ckpts/{model_name}_best-acc.pth"
        Path(save_path.replace("pth", "txt")).write_text(
            f"epoch-{epoch}_acc-{best_val_acc:.2f}",
            encoding="utf-8"
        )
        torch.save(model.state_dict(), save_path)

    if bad_epochs >= patience:
        break
