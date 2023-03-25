import zipfile
import os

data_dir_path = os.path.dirname(os.path.realpath(__file__))

def extract_zipfile(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    print(f'ZIP file {zip_path} extracted to {dest_dir}')
