from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _scan_recursive_samples(root, label_level):
    if label_level < 1:
        raise ValueError(f"label_level must be >= 1, got {label_level}")

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    samples = []
    too_shallow = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
            continue

        relative_dir = path.relative_to(root).parent.parts
        if len(relative_dir) < label_level:
            too_shallow.append(str(path))
            continue

        class_name = relative_dir[label_level - 1]
        samples.append((path, class_name))

    if too_shallow:
        preview = ', '.join(too_shallow[:3])
        raise RuntimeError(
            f"Found image files shallower than label_level={label_level} under {root}: {preview}"
        )

    if not samples:
        raise RuntimeError(
            f"No image files with supported extensions were found under {root}"
        )

    return samples


class RecursiveImageDataset(Dataset):
    def __init__(self, root, transform=None, label_level=1, class_to_idx=None):
        self.root = Path(root)
        self.transform = transform
        self.label_level = label_level

        raw_samples = _scan_recursive_samples(self.root, label_level)
        discovered_classes = sorted({class_name for _, class_name in raw_samples})

        if class_to_idx is None:
            self.class_to_idx = {name: idx for idx, name in enumerate(discovered_classes)}
        else:
            missing = sorted(set(discovered_classes) - set(class_to_idx))
            if missing:
                raise RuntimeError(
                    f"Classes under {self.root} are missing from the reference mapping: {missing}"
                )
            self.class_to_idx = dict(class_to_idx)

        self.classes = [
            name for name, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])
        ]
        self.samples = [(str(path), self.class_to_idx[class_name]) for path, class_name in raw_samples]
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            sample = image.copy()

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
