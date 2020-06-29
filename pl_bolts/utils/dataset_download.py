import os
import os.path
import gzip
import urllib
import tarfile
import zipfile
from tqdm import tqdm


def download_and_extract_archive(url, download_root, extract_root=None, filename=None):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    # lay out dir
    download_root = os.path.expanduser(download_root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(download_root, filename)
    os.makedirs(download_root, exist_ok=True)

    ext_name = url.split('/')[-1]
    file_downloaded_receipt = f'.{ext_name}.downloaded_receipt'
    receipt_path = os.path.join(download_root, file_downloaded_receipt)
    file_downloaded = os.path.exists(receipt_path)

    if not file_downloaded:
        # download first
        download_url(url, fpath)

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        extract_archive(archive, extract_root, remove_finished=True)

        # save the receipt instead of the full binary
        file = open(receipt_path, 'w')
        file.write(' ')
        file.close()

    num_downloaded_files = not file_downloaded
    return num_downloaded_files


def download_url(url, fpath):
    try:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(
            url, fpath,
            reporthook=gen_bar_updater()
        )
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        else:
            raise e


def extract_archive(from_path, to_path=None, remove_finished=False):  # pragma: no-cover
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")
