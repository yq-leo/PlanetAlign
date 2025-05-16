from typing import Union, Optional
from pathlib import Path
import os
import gdown

from PyNetAlign.data import Dataset


USER_AGENT = 'PyNetAlign'


def download_file_from_google_drive(
        remote_file_id: str,
        root: Union[str, Path],
        save_filename: Optional[Union[str, Path]] = None):
    r"""Download a file from Google Drive."""

    root = os.path.expanduser(root)
    if not save_filename:
        save_filename = remote_file_id
    fpath = os.fspath(os.path.join(root, save_filename))

    fname = save_filename.split('.')[0]
    os.makedirs(root, exist_ok=True)
    if os.path.isfile(fpath):
        print(f'Dataset "{fname}" is already downloaded')
        return
    else:
        print(f'Downloading dataset "{fname}"...')

    gdown.download(id=remote_file_id, output=fpath, quiet=False, user_agent=USER_AGENT)


if __name__ == '__main__':
    # Download the Phone-Email dataset
    download_file_from_google_drive(
        remote_file_id='13BklpBEFjT73Xk8H-daPGzBx-xFbpO0P',
        save_filename='phone_email.pt',
        root='../../datasets')

    phone_email = Dataset(root='../../datasets', name='phone_email', train_ratio=0.2)
    print(phone_email)

    # # Download the Douban dataset
    # download_file_from_google_drive(
    #     remote_file_id='1J5s9F9zv1Z6L6z4P3rWVlqfF7QvZ9w7R',
    #     root='../../datasets')
