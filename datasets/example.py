from PIL import Image
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms

class ExampleDataset(Dataset):

    def __init__(self, root="./datasets/example/", train=True,
                 transform=None):
        super().__init__()

        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"

        self.train = train
        self.sample_path = (
            self.root
            / (("train" if self.train else "val") + ".txt")
        )
        self.image_names = self.sample_path.read_text().splitlines()

        self.class_path = self.root / "classes.txt"
        self.classes = self.class_path.read_text().splitlines()
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        if train and transform is not None:
            self.transform.transforms.insert(1, transform)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]

        image_path = self.image_dir / image_name
        label_path = self.label_dir / image_name.replace("png", "txt")

        cls = Path(label_path).read_text().rstrip()

        image = self.transform(Image.open(image_path))
        label = self.class_to_id[cls]
        return image, label
