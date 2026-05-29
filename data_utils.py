import os
import glob
import cv2
import struct
import numpy as np
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

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
    mnist_save_path = './data/MNIST'
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

    # دقت کنید: ما دیگر y_train - 1 را انجام نمی‌دهیم تا لیبل‌ها 1 تا 9 باقی بمانند
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
    # لیبل‌ها دقیقاً همان مقادیر 1 تا 9 را خواهند داشت
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

def get_dataloaders(data_path, batch_size=128):
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
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

def get_dataloaders_mnist_hoda(data_path, batch_size=128):
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
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

def get_dataloaders_all(data_path, batch_size=128):
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
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader