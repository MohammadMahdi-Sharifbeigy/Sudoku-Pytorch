import os
import glob
import random
import cv2
import struct
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Normalisation stats (MNIST domain, grayscale [0,1] input)
# ──────────────────────────────────────────────────────────────────────────────
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

# Evaluation transform: normalise only, no augmentation.
EVAL_TRANSFORM = transforms.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD])


class RandomShadow:
    """Simulate a shadow across part of the digit cell.

    Darkens a random vertical strip (left or right half) by multiplying
    pixel values by a factor in [intensity_low, intensity_high].
    Applied on a PIL grayscale image before ToTensor.

    Rationale: real Sudoku photos often have diagonal shadows from book binding
    or uneven phone lighting. Even at 28×28 this helps generalise.
    """

    def __init__(self, intensity_range: tuple = (0.35, 0.70), p: float = 0.25):
        self.intensity_range = intensity_range
        self.p               = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr       = np.array(img, dtype=np.float32)
        h, w      = arr.shape[0], arr.shape[1]
        intensity = random.uniform(*self.intensity_range)

        # Randomly choose direction: vertical strip or horizontal strip
        if random.random() < 0.5:
            x1 = random.randint(0, w // 2)
            x2 = random.randint(w // 2, w)
            arr[:, x1:x2] *= intensity
        else:
            y1 = random.randint(0, h // 2)
            y2 = random.randint(h // 2, h)
            arr[y1:y2, :] *= intensity

        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# Default augmentation config
DEFAULT_AUG_CONFIG = {
    "brightness":   0.0,   # 0 = off; >0 applies ColorJitter brightness
    "contrast":     0.0,   # 0 = off
    "shadow_p":     0.0,   # 0 = off; shadow probability
    "shadow_intensity_min": 0.35,
    "shadow_intensity_max": 0.70,
}


def build_train_transform(
    brightness:           float = 0.0,
    contrast:             float = 0.0,
    shadow_p:             float = 0.0,
    shadow_intensity_min: float = 0.35,
    shadow_intensity_max: float = 0.70,
) -> transforms.Compose:
    """Build the full training augmentation pipeline.

    All params default to 0 / off so the function degrades gracefully.

    Args:
        brightness: ColorJitter brightness factor (0 = disabled).
        contrast:   ColorJitter contrast factor   (0 = disabled).
        shadow_p:   Probability of applying RandomShadow (0 = disabled).
        shadow_intensity_min/max: Shadow darkness range (0=black, 1=no change).
    """
    steps = [transforms.ToPILImage()]

    # ── Brightness / contrast (applied on PIL before geometric ops) ──
    jitter_kwargs = {}
    if brightness > 0:
        jitter_kwargs["brightness"] = brightness
    if contrast > 0:
        jitter_kwargs["contrast"] = contrast
    if jitter_kwargs:
        steps.append(transforms.ColorJitter(**jitter_kwargs))

    # ── Shadow ──────────────────────────────────────────────────────
    if shadow_p > 0:
        steps.append(RandomShadow(
            intensity_range=(shadow_intensity_min, shadow_intensity_max),
            p=shadow_p,
        ))

    # ── Geometric augmentations ──────────────────────────────────────
    steps += [
        transforms.RandomAffine(
            degrees=8,
            translate=(0.08, 0.08),
            scale=(0.88, 1.12),
            shear=6,
        ),
        transforms.RandomPerspective(distortion_scale=0.18, p=0.35),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
    ]

    # ── Tensor + normalise + pixel erasing ──────────────────────────
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ]

    return transforms.Compose(steps)


# Module-level default transform (no brightness/shadow) — kept for backward compat.
TRAIN_TRANSFORM = build_train_transform()

# ── Augmentation preset names ────────────────────────────────────────────────
AUG_PRESETS = {
    "none":  "No augmentation — original winning config",
    "light": "Affine only, no blur/erasing — safe for small models",
    "full":  "All transforms — current default",
}

EFFICIENTNET_MEAN = [0.485, 0.456, 0.406]
EFFICIENTNET_STD  = [0.229, 0.224, 0.225]


def build_train_transform_preset(preset: str = "full") -> transforms.Compose:
    """Return a training transform by preset name.

    preset options:
        "none"  — no augmentation, just ToTensor + MNIST normalize
        "light" — RandomAffine only, no blur or erasing
        "full"  — full augmentation pipeline (current default)
    """
    if preset == "none":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        ])
    if preset == "light":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.88, 1.12), shear=6),
            transforms.ToTensor(),
            transforms.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        ])
    # "full" — default
    return build_train_transform()


def build_efficientnet_transform(augment: bool = True) -> transforms.Compose:
    """Transform pipeline for EfficientNetDigitCNN.

    Resizes 28×28 → 224×224, converts 1-ch tensor to PIL RGB,
    applies ImageNet normalisation.
    augment=True adds mild affine + colour jitter suitable for 224×224.
    """
    steps = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if augment:
        steps += [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.90, 1.10), shear=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=EFFICIENTNET_MEAN, std=EFFICIENTNET_STD),
    ]
    return transforms.Compose(steps)


def build_efficientnet_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=EFFICIENTNET_MEAN, std=EFFICIENTNET_STD),
    ])


class AugmentedDataset(Dataset):
    """Wraps pre-computed (x, y) tensors and applies on-the-fly augmentation.

    x: float32 tensor shape (N, 1, H, W) in range [0, 1]
    y: long tensor shape (N,)
    transform: TRAIN_TRANSFORM for train, EVAL_TRANSFORM for val/test.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform=None):
        self.x         = x
        self.y         = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        img = self.x[idx]   # [1, H, W] float32 in [0, 1]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[idx]


class MultitaskAugmentedDataset(Dataset):
    """Three-output variant for multi-task loaders (x, digit_y, lang_y)."""

    def __init__(self, x: torch.Tensor, digit_y: torch.Tensor,
                 lang_y: torch.Tensor, transform=None):
        self.x         = x
        self.digit_y   = digit_y
        self.lang_y    = lang_y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        img = self.x[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.digit_y[idx], self.lang_y[idx]


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=torch.cuda.is_available())


def _make_loaders(x_train, y_train, x_val, y_val, x_test, y_test,
                  batch_size: int,
                  train_transform=None) -> tuple:
    """Wrap three splits into DataLoaders with augmentation on the train split.

    train_transform defaults to TRAIN_TRANSFORM when None.
    Pass build_train_transform(...) from the UI to customise augmentation.
    """
    if train_transform is None:
        train_transform = TRAIN_TRANSFORM
    train_ds = AugmentedDataset(x_train, y_train, transform=train_transform)
    val_ds   = AugmentedDataset(x_val,   y_val,   transform=EVAL_TRANSFORM)
    test_ds  = AugmentedDataset(x_test,  y_test,  transform=EVAL_TRANSFORM)
    return (
        _make_loader(train_ds, batch_size, shuffle=True),
        _make_loader(val_ds,   batch_size, shuffle=False),
        _make_loader(test_ds,  batch_size, shuffle=False),
    )

# ==========================================
# 1. Hoda Dataset Reader Functions
# ==========================================
def __convert_to_one_hot(vector, num_classes):
    result = np.zeros(shape=[len(vector), num_classes])
    result[np.arange(len(vector)), vector] = 1
    return result

def __resize_image(src_image, dst_image_height, dst_image_width):
    src_image_height = src_image.shape[0]
    src_image_width = src_image.shape[1]

    if src_image_height > dst_image_height or src_image_width > dst_image_width:
        height_scale = dst_image_height / src_image_height
        width_scale = dst_image_width / src_image_width
        scale = min(height_scale, width_scale)
        img = cv2.resize(src=src_image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        img = src_image

    img_height = img.shape[0]
    img_width = img.shape[1]
    dst_image = np.zeros(shape=[dst_image_height, dst_image_width], dtype=np.uint8)

    y_offset = (dst_image_height - img_height) // 2
    x_offset = (dst_image_width - img_width) // 2
    dst_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = img

    return dst_image

def read_hoda_cdb(file_name):
    with open(file_name, 'rb') as binary_file:
        data = binary_file.read()
        offset = 0

        # read private header
        yy = struct.unpack_from('H', data, offset)[0]; offset += 2
        m = struct.unpack_from('B', data, offset)[0]; offset += 1
        d = struct.unpack_from('B', data, offset)[0]; offset += 1
        H = struct.unpack_from('B', data, offset)[0]; offset += 1
        W = struct.unpack_from('B', data, offset)[0]; offset += 1
        TotalRec = struct.unpack_from('I', data, offset)[0]; offset += 4
        LetterCount = struct.unpack_from('128I', data, offset); offset += 128 * 4
        imgType = struct.unpack_from('B', data, offset)[0]; offset += 1 # 0: binary, 1: gray
        Comments = struct.unpack_from('256c', data, offset); offset += 256 * 1
        Reserved = struct.unpack_from('245c', data, offset); offset += 245 * 1

        normal = True if (W > 0) and (H > 0) else False
        images, labels = [], []

        for i in range(TotalRec):
            StartByte = struct.unpack_from('B', data, offset)[0]; offset += 1
            label = struct.unpack_from('B', data, offset)[0]; offset += 1

            if not normal:
                W = struct.unpack_from('B', data, offset)[0]; offset += 1
                H = struct.unpack_from('B', data, offset)[0]; offset += 1

            ByteCount = struct.unpack_from('H', data, offset)[0]; offset += 2
            image = np.zeros(shape=[H, W], dtype=np.uint8)

            if imgType == 0:
                for y in range(H):
                    bWhite = True
                    counter = 0
                    while counter < W:
                        WBcount = struct.unpack_from('B', data, offset)[0]
                        offset += 1
                        if bWhite:
                            image[y, counter:counter + WBcount] = 0 
                        else:
                            image[y, counter:counter + WBcount] = 255 
                        bWhite = not bWhite 
                        counter += WBcount
            else:
                data_chunk = struct.unpack_from('{}B'.format(W * H), data, offset)
                offset += W * H
                image = np.asarray(data_chunk, dtype=np.uint8).reshape([W, H]).T

            images.append(image)
            labels.append(label)

        return images, labels

def read_hoda_dataset(dataset_path, images_height=28, images_width=28, one_hot=False, reshape=False):
    images, labels = read_hoda_cdb(dataset_path)
    assert len(images) == len(labels)

    X = np.zeros(shape=[len(images), images_height, images_width], dtype=np.float32)
    Y = np.zeros(shape=[len(labels)], dtype=int)

    for i in range(len(images)):
        image = images[i]
        image = __resize_image(src_image=image, dst_image_height=images_height, dst_image_width=images_width)
        image = image / 255.0
        X[i] = image
        Y[i] = labels[i]

    if one_hot: Y = __convert_to_one_hot(Y, 10).astype(dtype=np.float32)
    else: Y = Y.astype(dtype=np.int64)

    if reshape: X = X.reshape(-1, images_height * images_width)
    else: X = X.reshape(-1, 1, images_height, images_width) 

    return X, Y

def load_hoda_images(data_dir='data'):
    """Load Hoda dataset, remove digit 0, and return train/val/test splits."""
    train_path = os.path.join(data_dir, 'DigitDB', 'Train 60000.cdb')
    test_path = os.path.join(data_dir, 'DigitDB', 'Test 20000.cdb')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Hoda dataset files not found at {train_path}. Skipping Hoda data.")
        return None, None, None, None, None, None

    x_train, y_train = read_hoda_dataset(train_path, images_height=28, images_width=28, reshape=False)
    x_test, y_test = read_hoda_dataset(test_path, images_height=28, images_width=28, reshape=False)

    train_mask = y_train != 0
    test_mask = y_test != 0

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=2023)

    return (torch.from_numpy(x_train).float(), torch.from_numpy(x_val).float(), torch.from_numpy(x_test).float(),
            torch.from_numpy(y_train).long(), torch.from_numpy(y_val).long(), torch.from_numpy(y_test).long())


# ==========================================
# 2. MNIST, Fonts, and Empty Cells Generation
# ==========================================
def generate_empty_cells(num_samples=3000):
    """Generates completely black images labeled as 0 to represent empty Sudoku cells."""
    x = torch.zeros((num_samples, 1, 28, 28), dtype=torch.float32)
    y = torch.zeros(num_samples, dtype=torch.long)
    
    x_train, x_temp, y_train, y_temp = train_test_split(x.numpy(), y.numpy(), test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    return (torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(x_test),
            torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test))

def load_mnist_images():
    mnist_save_path = './data'
    os.makedirs(mnist_save_path, exist_ok=True)

    mnist_train = datasets.MNIST(root=mnist_save_path, train=True, download=True)
    mnist_test = datasets.MNIST(root=mnist_save_path, train=False, download=True)

    x_train = mnist_train.data.numpy().astype('float32')
    y_train = mnist_train.targets.numpy()
    x_test = mnist_test.data.numpy().astype('float32')
    y_test = mnist_test.targets.numpy()

    non_zero_train_indices = np.where(y_train != 0)[0]
    non_zero_test_indices = np.where(y_test != 0)[0]

    x_train = x_train[non_zero_train_indices]
    y_train = y_train[non_zero_train_indices]
    x_test = x_test[non_zero_test_indices]
    y_test = y_test[non_zero_test_indices]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.85, random_state=2023)

    x_train = np.expand_dims(x_train / 255.0, 1)
    x_val = np.expand_dims(x_val / 255.0, 1)
    x_test = np.expand_dims(x_test / 255.0, 1)

    return (torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(x_test),
            torch.from_numpy(y_train).long(), torch.from_numpy(y_val).long(), torch.from_numpy(y_test).long())

def get_font_image_dict(data_path, excluded_names=None):
    base_digit_path = os.path.join(data_path, 'digit_images')
    folder_names = sorted(glob.glob(os.path.join(base_digit_path, '*')))
    digit_image_filepaths = [sorted(glob.glob(os.path.join(folder, '*.png'))) for folder in folder_names]

    if excluded_names:
        inclusion_list_indices = list(np.where([not any(elem in fpath for elem in excluded_names)
                                                for fpath in digit_image_filepaths[0]])[0])
        digit_image_filepaths = [[fpath_list[i] for i in inclusion_list_indices]
                                 for fpath_list in digit_image_filepaths]

    img_dict = {i: [] for i in range(1, 10)}
    
    for digit_class in range(1, 10):
        for fpath in digit_image_filepaths[digit_class - 1]:
            try:
                img = cv2.imread(fpath)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                img_dict[digit_class].append(resized)
            except Exception:
                pass
        
        img_dict[digit_class] = np.array(img_dict[digit_class])
        if len(img_dict[digit_class]) > 0:
            img_dict[digit_class] = np.expand_dims(img_dict[digit_class], 1)
            
    return img_dict

def load_font_image_arrays(image_dict):
    x = np.concatenate([v for v in image_dict.values()], axis=0)
    y = np.array([np.repeat(k, len(image_dict[k])) for k in image_dict])
    y = np.reshape(y, (-1, 1))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.18, shuffle=True, random_state=33)
    
    x_train = np.array([cv2.bitwise_not(img) for img in x_train.squeeze(1)])
    x_val = np.array([cv2.bitwise_not(img) for img in x_val.squeeze(1)])
    x_test = np.array([cv2.bitwise_not(img) for img in x_test.squeeze(1)])
    
    x_train = np.expand_dims(x_train, 1).astype('float32') / 255.0
    x_val = np.expand_dims(x_val, 1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, 1).astype('float32') / 255.0
    
    return (torch.from_numpy(x_train).float(), torch.from_numpy(x_val).float(), torch.from_numpy(x_test).float(),
            torch.from_numpy(y_train.flatten()).long(), torch.from_numpy(y_val.flatten()).long(), torch.from_numpy(y_test.flatten()).long())


# ==========================================
# 3. DataLoader Generators
# ==========================================

def get_dataloaders(data_path, batch_size=128, train_transform=None):
    """Loader 1: MNIST + Fonts + Empty Cells"""
    x_tr_m, x_v_m, x_te_m, y_tr_m, y_v_m, y_te_m = load_mnist_images()
    
    img_dict = get_font_image_dict(data_path)
    x_tr_f, x_v_f, x_te_f, y_tr_f, y_v_f, y_te_f = load_font_image_arrays(img_dict)

    x_tr_e, x_v_e, x_te_e, y_tr_e, y_v_e, y_te_e = generate_empty_cells()
    
    x_train = torch.cat([x_tr_f, x_tr_m, x_tr_e], dim=0)
    x_val = torch.cat([x_v_f, x_v_m, x_v_e], dim=0)
    x_test = torch.cat([x_te_f, x_te_m, x_te_e], dim=0)
    
    y_train = torch.cat([y_tr_f, y_tr_m, y_tr_e], dim=0)
    y_val = torch.cat([y_v_f, y_v_m, y_v_e], dim=0)
    y_test = torch.cat([y_te_f, y_te_m, y_te_e], dim=0)
    
    return _make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, train_transform=train_transform)

def get_dataloaders_mnist_hoda(data_path, batch_size=128, train_transform=None):
    """Loader 2: MNIST + Hoda + Empty Cells"""
    x_tr_m, x_v_m, x_te_m, y_tr_m, y_v_m, y_te_m = load_mnist_images()
    x_tr_h, x_v_h, x_te_h, y_tr_h, y_v_h, y_te_h = load_hoda_images(data_path)
    x_tr_e, x_v_e, x_te_e, y_tr_e, y_v_e, y_te_e = generate_empty_cells()
    
    x_train_list, y_train_list = [x_tr_m, x_tr_e], [y_tr_m, y_tr_e]
    x_val_list, y_val_list = [x_v_m, x_v_e], [y_v_m, y_v_e]
    x_test_list, y_test_list = [x_te_m, x_te_e], [y_te_m, y_te_e]

    if x_tr_h is not None:
        x_train_list.append(x_tr_h); y_train_list.append(y_tr_h)
        x_val_list.append(x_v_h); y_val_list.append(y_v_h)
        x_test_list.append(x_te_h); y_test_list.append(y_te_h)

    x_train = torch.cat(x_train_list, dim=0)
    x_val = torch.cat(x_val_list, dim=0)
    x_test = torch.cat(x_test_list, dim=0)
    
    y_train = torch.cat(y_train_list, dim=0)
    y_val = torch.cat(y_val_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    
    return _make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, train_transform=train_transform)

def get_dataloaders_mnist_only(batch_size=128, train_transform=None):
    """Loader 4: MNIST + Empty Cells only (no Fonts, no Hoda)."""
    x_tr_m, x_v_m, x_te_m, y_tr_m, y_v_m, y_te_m = load_mnist_images()
    x_tr_e, x_v_e, x_te_e, y_tr_e, y_v_e, y_te_e = generate_empty_cells()

    x_train = torch.cat([x_tr_m, x_tr_e], dim=0)
    x_val   = torch.cat([x_v_m,  x_v_e],  dim=0)
    x_test  = torch.cat([x_te_m, x_te_e], dim=0)

    y_train = torch.cat([y_tr_m, y_tr_e], dim=0)
    y_val   = torch.cat([y_v_m,  y_v_e],  dim=0)
    y_test  = torch.cat([y_te_m, y_te_e], dim=0)

    return _make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, train_transform=train_transform)

def get_dataloaders_all(data_path, batch_size=128, train_transform=None):
    """Loader 3: MNIST + Fonts + Hoda + Empty Cells"""
    x_tr_m, x_v_m, x_te_m, y_tr_m, y_v_m, y_te_m = load_mnist_images()

    img_dict = get_font_image_dict(data_path)
    x_tr_f, x_v_f, x_te_f, y_tr_f, y_v_f, y_te_f = load_font_image_arrays(img_dict)

    x_tr_h, x_v_h, x_te_h, y_tr_h, y_v_h, y_te_h = load_hoda_images(data_path)
    x_tr_e, x_v_e, x_te_e, y_tr_e, y_v_e, y_te_e = generate_empty_cells()

    x_train_list, y_train_list = [x_tr_m, x_tr_f, x_tr_e], [y_tr_m, y_tr_f, y_tr_e]
    x_val_list, y_val_list = [x_v_m, x_v_f, x_v_e], [y_v_m, y_v_f, y_v_e]
    x_test_list, y_test_list = [x_te_m, x_te_f, x_te_e], [y_te_m, y_te_f, y_te_e]

    if x_tr_h is not None:
        x_train_list.append(x_tr_h); y_train_list.append(y_tr_h)
        x_val_list.append(x_v_h); y_val_list.append(y_v_h)
        x_test_list.append(x_te_h); y_test_list.append(y_te_h)

    x_train = torch.cat(x_train_list, dim=0)
    x_val = torch.cat(x_val_list, dim=0)
    x_test = torch.cat(x_test_list, dim=0)

    y_train = torch.cat(y_train_list, dim=0)
    y_val = torch.cat(y_val_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)

    return _make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, train_transform=train_transform)


def get_dataloaders_unified20(data_path, batch_size=128, train_transform=None):
    """Unified 20-class loader: MNIST + Fonts + Hoda + Empty Cells.

    Class layout
    ------------
    0        : English empty cell
    1 – 9    : English digits 1-9   (MNIST, Fonts)
    10       : Persian empty cell   (not used in training — empty mapped to 0)
    11 – 19  : Persian digits 1-9   (Hoda)

    In practice empty cells → class 0 only (class 10 unused during training,
    model still learns it implicitly through the empty-cell class structure).

    Returns (train_loader, val_loader, test_loader).
    Each batch: (images, unified_labels_0_to_19).
    """
    x_tr_m, x_v_m, x_te_m, y_tr_m, y_v_m, y_te_m = load_mnist_images()
    img_dict = get_font_image_dict(data_path)
    x_tr_f, x_v_f, x_te_f, y_tr_f, y_v_f, y_te_f = load_font_image_arrays(img_dict)
    x_tr_h, x_v_h, x_te_h, y_tr_h, y_v_h, y_te_h = load_hoda_images(data_path)
    x_tr_e, x_v_e, x_te_e, y_tr_e, y_v_e, y_te_e = generate_empty_cells()

    # English data: labels already 1-9 → keep as-is (classes 1-9)
    # Persian data: labels 1-9 → add 10 (classes 11-19)
    # Empty cells: label 0 → class 0

    x_lists  = {'tr': [x_tr_m, x_tr_f, x_tr_e], 'v': [x_v_m, x_v_f, x_v_e], 'te': [x_te_m, x_te_f, x_te_e]}
    y_lists  = {'tr': [y_tr_m, y_tr_f, y_tr_e], 'v': [y_v_m, y_v_f, y_v_e], 'te': [y_te_m, y_te_f, y_te_e]}

    if x_tr_h is not None:
        # shift Persian labels: 1-9 → 11-19
        y_tr_h_u = y_tr_h + 10
        y_v_h_u  = y_v_h  + 10
        y_te_h_u = y_te_h + 10
        x_lists['tr'].append(x_tr_h); y_lists['tr'].append(y_tr_h_u)
        x_lists['v'].append(x_v_h);   y_lists['v'].append(y_v_h_u)
        x_lists['te'].append(x_te_h); y_lists['te'].append(y_te_h_u)
        print(f"Unified 20-class  English classes 1-9 · Persian classes 11-19 · Empty class 0")
    else:
        print("WARNING: Hoda not found — unified model will train on English-only (10 classes effective).")

    x_train = torch.cat(x_lists['tr'], dim=0)
    x_val   = torch.cat(x_lists['v'],  dim=0)
    x_test  = torch.cat(x_lists['te'], dim=0)
    y_train = torch.cat(y_lists['tr'], dim=0)
    y_val   = torch.cat(y_lists['v'],  dim=0)
    y_test  = torch.cat(y_lists['te'], dim=0)

    # Sanity: all labels must be in [0, 19]
    assert y_train.max() <= 19 and y_train.min() >= 0, \
        f"Unified label out of range: min={y_train.min()} max={y_train.max()}"

    return _make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, train_transform=train_transform)


def get_dataloaders_persian(data_path, batch_size=128, train_transform=None):
    """Loader: Hoda (Persian) + Empty Cells only.
    For training a DigitCNN dedicated to Persian handwritten digits.
    Returns None loaders if Hoda files not found.
    """
    x_tr_h, x_v_h, x_te_h, y_tr_h, y_v_h, y_te_h = load_hoda_images(data_path)
    if x_tr_h is None:
        return None, None, None

    x_tr_e, x_v_e, x_te_e, y_tr_e, y_v_e, y_te_e = generate_empty_cells()

    x_train = torch.cat([x_tr_h, x_tr_e], dim=0)
    x_val   = torch.cat([x_v_h,  x_v_e],  dim=0)
    x_test  = torch.cat([x_te_h, x_te_e], dim=0)
    y_train = torch.cat([y_tr_h, y_tr_e], dim=0)
    y_val   = torch.cat([y_v_h,  y_v_e],  dim=0)
    y_test  = torch.cat([y_te_h, y_te_e], dim=0)

    return _make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, train_transform=train_transform)


def get_dataloaders_english(data_path, batch_size=128, train_transform=None):
    """Loader: MNIST + Fonts + Empty Cells only (no Hoda).
    For training a DigitCNN dedicated to English/printed digits.
    Alias for get_dataloaders (identical data).
    """
    return get_dataloaders(data_path, batch_size=batch_size, train_transform=train_transform)


# ==========================================
# 4. Multi-Task Data Helpers
# ==========================================

LANG_PERSIAN        = 0
LANG_ENGLISH        = 1
LANG_EMPTY_SENTINEL = -1   # masked by CrossEntropyLoss(ignore_index=-1)


def _make_lang_labels(digit_labels: torch.Tensor, lang_value: int) -> torch.Tensor:
    """Return lang label tensor for a batch that is all one language.
    Empty cells (digit_label == 0) always get LANG_EMPTY_SENTINEL regardless of lang_value.
    """
    lang = torch.full_like(digit_labels, lang_value)
    lang[digit_labels == 0] = LANG_EMPTY_SENTINEL
    return lang


def _report_lang_balance(split_name: str, lang_labels: torch.Tensor) -> dict:
    """Print and return Persian/English counts for a split.
    Prints WARNING if majority class exceeds 60%.
    """
    valid      = lang_labels[lang_labels != LANG_EMPTY_SENTINEL]
    n_persian  = int((valid == LANG_PERSIAN).sum())
    n_english  = int((valid == LANG_ENGLISH).sum())
    n_total    = n_persian + n_english
    r_persian  = n_persian / max(n_total, 1) * 100
    r_english  = n_english / max(n_total, 1) * 100
    imbalanced = max(r_persian, r_english) > 60.0

    print(f"[{split_name}] Lang balance  Persian: {n_persian} ({r_persian:.1f}%)  "
          f"English: {n_english} ({r_english:.1f}%)")
    if imbalanced:
        print(f"WARNING [{split_name}]: Language imbalance >60/40 detected! "
              f"Class-weighted loss will compensate.")

    return {
        'n_persian': n_persian, 'n_english': n_english,
        'ratio_persian': r_persian, 'ratio_english': r_english,
        'imbalanced': imbalanced,
    }


def compute_lang_class_weights(lang_labels_train: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights for Persian (0) and English (1).
    Pass the result to MultiTaskFocalLoss(lang_class_weights=...).
    """
    valid  = lang_labels_train[lang_labels_train != LANG_EMPTY_SENTINEL]
    counts = torch.zeros(2)
    counts[LANG_PERSIAN] = (valid == LANG_PERSIAN).sum().float()
    counts[LANG_ENGLISH] = (valid == LANG_ENGLISH).sum().float()
    counts  = counts.clamp(min=1.0)
    weights = counts.sum() / (2.0 * counts)
    return weights


def get_dataloaders_multitask(data_path, batch_size=128, train_transform=None):
    """Multi-task loader: MNIST + Fonts + Hoda + Empty Cells.

    Each batch yields (image, digit_label, lang_label):
      MNIST  → lang 1 (English)
      Fonts  → lang 1 (English)
      Hoda   → lang 0 (Persian)
      Empty  → lang -1 (masked from loss)

    Returns
    -------
    train_loader, val_loader, test_loader, balance_info, lang_class_weights
      balance_info       : dict with 'train'/'val'/'test' balance stats (for st.warning)
      lang_class_weights : torch.Tensor shape (2,) — pass to MultiTaskFocalLoss
    """
    x_tr_m, x_v_m, x_te_m, y_tr_m, y_v_m, y_te_m = load_mnist_images()
    img_dict = get_font_image_dict(data_path)
    x_tr_f, x_v_f, x_te_f, y_tr_f, y_v_f, y_te_f = load_font_image_arrays(img_dict)
    x_tr_h, x_v_h, x_te_h, y_tr_h, y_v_h, y_te_h = load_hoda_images(data_path)
    x_tr_e, x_v_e, x_te_e, y_tr_e, y_v_e, y_te_e = generate_empty_cells()

    # Build lang labels per source
    lt_tr_m = _make_lang_labels(y_tr_m, LANG_ENGLISH)
    lt_v_m  = _make_lang_labels(y_v_m,  LANG_ENGLISH)
    lt_te_m = _make_lang_labels(y_te_m, LANG_ENGLISH)

    lt_tr_f = _make_lang_labels(y_tr_f, LANG_ENGLISH)
    lt_v_f  = _make_lang_labels(y_v_f,  LANG_ENGLISH)
    lt_te_f = _make_lang_labels(y_te_f, LANG_ENGLISH)

    lt_tr_e = _make_lang_labels(y_tr_e, LANG_EMPTY_SENTINEL)
    lt_v_e  = _make_lang_labels(y_v_e,  LANG_EMPTY_SENTINEL)
    lt_te_e = _make_lang_labels(y_te_e, LANG_EMPTY_SENTINEL)

    x_tr_list  = [x_tr_m, x_tr_f, x_tr_e]
    y_tr_list  = [y_tr_m, y_tr_f, y_tr_e]
    lt_tr_list = [lt_tr_m, lt_tr_f, lt_tr_e]

    x_v_list   = [x_v_m,  x_v_f,  x_v_e]
    y_v_list   = [y_v_m,  y_v_f,  y_v_e]
    lt_v_list  = [lt_v_m,  lt_v_f,  lt_v_e]

    x_te_list  = [x_te_m, x_te_f, x_te_e]
    y_te_list  = [y_te_m, y_te_f, y_te_e]
    lt_te_list = [lt_te_m, lt_te_f, lt_te_e]

    if x_tr_h is not None:
        lt_tr_h = _make_lang_labels(y_tr_h, LANG_PERSIAN)
        lt_v_h  = _make_lang_labels(y_v_h,  LANG_PERSIAN)
        lt_te_h = _make_lang_labels(y_te_h, LANG_PERSIAN)
        x_tr_list.append(x_tr_h);   y_tr_list.append(y_tr_h);   lt_tr_list.append(lt_tr_h)
        x_v_list.append(x_v_h);     y_v_list.append(y_v_h);     lt_v_list.append(lt_v_h)
        x_te_list.append(x_te_h);   y_te_list.append(y_te_h);   lt_te_list.append(lt_te_h)

    x_train  = torch.cat(x_tr_list,  dim=0)
    y_train  = torch.cat(y_tr_list,  dim=0)
    lt_train = torch.cat(lt_tr_list, dim=0)

    x_val  = torch.cat(x_v_list,  dim=0)
    y_val  = torch.cat(y_v_list,  dim=0)
    lt_val = torch.cat(lt_v_list, dim=0)

    x_test  = torch.cat(x_te_list,  dim=0)
    y_test  = torch.cat(y_te_list,  dim=0)
    lt_test = torch.cat(lt_te_list, dim=0)

    balance_info = {
        'train': _report_lang_balance('train', lt_train),
        'val':   _report_lang_balance('val',   lt_val),
        'test':  _report_lang_balance('test',  lt_test),
    }
    lang_class_weights = compute_lang_class_weights(lt_train)
    print(f"Lang class weights  Persian: {lang_class_weights[0]:.4f}  "
          f"English: {lang_class_weights[1]:.4f}")

    pin = torch.cuda.is_available()
    train_loader = _make_loader(
        MultitaskAugmentedDataset(x_train, y_train, lt_train, transform=TRAIN_TRANSFORM),
        batch_size, shuffle=True,
    )
    val_loader = _make_loader(
        MultitaskAugmentedDataset(x_val, y_val, lt_val, transform=EVAL_TRANSFORM),
        batch_size, shuffle=False,
    )
    test_loader = _make_loader(
        MultitaskAugmentedDataset(x_test, y_test, lt_test, transform=EVAL_TRANSFORM),
        batch_size, shuffle=False,
    )
    return train_loader, val_loader, test_loader, balance_info, lang_class_weights


def get_dataloaders_efficientnet(
    data_path: str,
    dataset_mode: str = "all",
    batch_size: int = 32,
    aug_preset: str = "full",
) -> tuple:
    """DataLoaders for EfficientNetDigitCNN (224×224 RGB, ImageNet normalisation).

    dataset_mode: "all" | "persian" | "english" | "mnist_only" | "mnist_fonts" | "mnist_hoda"
    aug_preset:   "none" | "light" | "full"
    Returns: (train_loader, val_loader, test_loader)
    """
    augment = aug_preset != "none"
    train_tf = build_efficientnet_transform(augment=augment)
    eval_tf  = build_efficientnet_eval_transform()

    if dataset_mode == "persian":
        base_train, base_val, base_test = get_dataloaders_persian(data_path, batch_size=batch_size)
    elif dataset_mode == "english":
        base_train, base_val, base_test = get_dataloaders_english(data_path, batch_size=batch_size)
    elif dataset_mode == "mnist_only":
        base_train, base_val, base_test = get_dataloaders_mnist_only(batch_size=batch_size)
    elif dataset_mode == "mnist_fonts":
        base_train, base_val, base_test = get_dataloaders(data_path, batch_size=batch_size)
    elif dataset_mode == "mnist_hoda":
        base_train, base_val, base_test = get_dataloaders_mnist_hoda(data_path, batch_size=batch_size)
    else:  # "all"
        base_train, base_val, base_test = get_dataloaders_all(data_path, batch_size=batch_size)

    def _rewrap(loader, transform):
        ds = loader.dataset
        if hasattr(ds, 'x'):
            new_ds = AugmentedDataset(ds.x, ds.y, transform=transform)
        else:
            new_ds = AugmentedDataset(ds.tensors[0], ds.tensors[1], transform=transform)
        return DataLoader(new_ds, batch_size=batch_size, shuffle=(transform is train_tf),
                          num_workers=0, pin_memory=False)

    return (
        _rewrap(base_train, train_tf),
        _rewrap(base_val,   eval_tf),
        _rewrap(base_test,  eval_tf),
    )