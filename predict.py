from datasets.example import ExampleDataset
from models.example import ExampleModel

from PIL import Image
from pathlib import Path

import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ExampleDataset()

class_to_idx = dataset.class_to_id
id_to_class = { v: k for k, v in class_to_idx.items() }

model = ExampleModel()


model.load_state_dict(
    torch.load(
        "./ckpts/ExampleModel_best-acc.pth",
        map_location=device,
        weights_only=True,
    )
)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

model.eval()
with torch.inference_mode():

    image_path = Path("./datasets/example/images/cifar_test_batch_06230.png")

    image = Image.open(image_path)

    x = transform(image)
    x = x.reshape(-1, 3, 32, 32).to(device)

    y = model(x)
    cls = id_to_class[torch.argmax(y).item()]

    print(cls)
