import os
import shutil
import random

# Đường dẫn tới thư mục chứa dữ liệu
data_dir = 'dataset1/photo_train'
photo_dir = os.path.join(data_dir, 'photo/tx_000000000000')
sketch_dir = os.path.join(data_dir, 'sketch/tx_000000000000')

# # tạo thư mục dataset chứa train và test
os.makedirs('dataset1/photo_val', exist_ok=True)
os.makedirs('dataset1/photo_val', exist_ok=True)
dataset_dir = 'dataset'
photo_train_dir = os.path.join(dataset_dir, 'photo_train')
sketch_train_dir = os.path.join(dataset_dir, 'sketch_train')
photo_test_dir = os.path.join(dataset_dir, 'photo_test')
sketch_test_dir = os.path.join(dataset_dir, 'sketch_test')

os.makedirs(photo_train_dir, exist_ok=True)
os.makedirs(sketch_train_dir, exist_ok=True)
os.makedirs(photo_test_dir, exist_ok=True)
os.makedirs(sketch_test_dir, exist_ok=True)

def create_train_test(x, y):
    category_dir = os.path.join(x, y)
    os.makedirs(category_dir, exist_ok=True)

# Lặp qua từng category
for category_folder in os.listdir(photo_dir):
    create_train_test(photo_train_dir, category_folder)
    create_train_test(sketch_train_dir, category_folder)
    create_train_test(photo_test_dir, category_folder)
    create_train_test(sketch_test_dir, category_folder)

# Tỷ lệ phân chia train và test
train_ratio = 0.8


for category_folder in os.listdir(photo_dir):
    category_photo_path = os.path.join(photo_dir, category_folder)
    category_sketch_path = os.path.join(sketch_dir, category_folder)
    
    # Lấy danh sách các ảnh trong category
    photos = os.listdir(category_photo_path)
    
    # Chia dữ liệu thành train và test
    random.shuffle(photos)
    train_photos = photos[:int(len(photos) * train_ratio)]
    test_photos = photos[int(len(photos) * train_ratio):]
    
    train_photo_path = os.path.join(photo_train_dir, category_folder)
    test_photo_path = os.path.join(photo_test_dir, category_folder)

    train_sketch_path = os.path.join(sketch_train_dir, category_folder)
    test_sketch_path = os.path.join(sketch_test_dir, category_folder)

    # Di chuyển ảnh vào thư mục train và test tương ứng
    for photo in train_photos:
        shutil.copy(os.path.join(category_photo_path, photo), train_photo_path)
    for photo in test_photos:
        shutil.copy(os.path.join(category_photo_path, photo), test_photo_path)
    
    sketchs = os.listdir(category_sketch_path)

    # Copy các sketch train và test tương ứng
    for photo in train_photos:
        path_photo = photo.split('/')
        photo_name = path_photo[-1].split(".")[0]
        for i in range(len(sketchs)):
            sketch = sketchs[i]
            path = sketch.split('/')
            sketch_name = path[-1].split("-")[0]
            if sketch_name == photo_name:
                shutil.copy(os.path.join(category_sketch_path, sketch), train_sketch_path)

    for photo in test_photos:
        path_photo = photo.split('/')
        photo_name = path_photo[-1].split(".")[0]
        for i in range(len(sketchs)):
            sketch = sketchs[i]
            path = sketch.split('/')
            sketch_name = path[-1].split("-")[0]
            if sketch_name == photo_name:
                shutil.copy(os.path.join(category_sketch_path, sketch), test_sketch_path)