import os
import shutil

src_dir = './JPG'
raw_dir = './raw'
new_dir = './uitgezocht'

# make new dir if it doesn't exist
os.makedirs(new_dir, exist_ok=True)
print('folder created or already exists')

# get all the filenames
src_files = os.listdir(src_dir)
raw_files = [f"{filename.split('.')[0]}.cr3" for filename in src_files]

# copy the raw_files into new_dir if they exist
for file in raw_files:
    src_path = os.path.join(raw_dir, file)
    dst_path = os.path.join(new_dir, file)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {file}")
    else:
        print(f"Missing: {file}")


