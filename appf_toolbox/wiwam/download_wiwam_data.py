# from bs4 import BeautifulSoup
# import requests
# from tqdm import tqdm
# from pathlib import Path
# import urllib.request
# import pandas as pd
#
# df = pd.read_csv("/media/huajian/Files/python_projects/chickpea_ali_montana_0664/pippa_info_0664 chickpea_merged.csv")
# local_base_dir = '/media/huajian/TOSHIBA EXT/chickpea_ali_0664'
#
# for r, row in df.iterrows():
#
#     for url in [f"https://projects.pawsey.org.au/appf-quick-data-sharing?prefix=0664_hyperspec/swir{row['image_url'][4:]}",
#                f"https://projects.pawsey.org.au/appf-quick-data-sharing?prefix=0664_hyperspec/vnir{row['image_url'][4:]}"]:
#
#
#         page = requests.get(url)
#         soup = BeautifulSoup(page.content, features="html")
#
#         keys = soup.find_all('key')
#         print(url)
#         print(keys)
#
#         for key in tqdm([k for k in keys if not k.get_text()[-1] == '/']):
#             local = Path(f'{local_base_dir}/{key.get_text()}')
#             local.parent.mkdir(parents=True, exist_ok=True)
#
#             urllib.request.urlretrieve(f"{url}/{key.get_text().replace(' ','%20')}", local)

import boto3
import os
import pandas as pd

pippa_file = 'E:/python_projects/chickpea_ali_montana_0664/pippa_info_0664 chickpea_merged.csv'
save_path = 'D:/chickpea_ali_0664'

#s3 = boto3.resource('s3') # assumes credentials & configuration are handled outside python in .aws directory or environment variables

s3 = boto3.resource(service_name='s3',
                 aws_access_key_id = 'xxx',
                 aws_secret_access_key='xxx',
                  endpoint_url = 'https://projects.pawsey.org.au')

BUCKET_NAME = 'appf-tpa-wiwam-datastore'

df = pd.read_csv(pippa_file)
local_base_dir = save_path

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

for r, row in df.iterrows():
    for camera in ["swir","vnir"]:
        filename = camera+row['image_url'][4:]
        download_s3_folder(BUCKET_NAME, filename, f"{local_base_dir}/{filename}")