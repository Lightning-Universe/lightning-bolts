import os
from pl_bolts.datamodules import LightningDataModule


def test_data_download(tmpdir):
    # --------------------------
    # test that when file isn't there it is downloaded
    # --------------------------
    dm = LightningDataModule(tmpdir)
    dm.add_data_url('https://pl-public-data.s3.amazonaws.com/test_0.tar.gz')
    num_urls_downloaded = dm.download_registered_data_urls()

    # we only downloaded one file
    assert num_urls_downloaded == 1

    files = os.listdir(tmpdir)
    assert '.test_0.tar.gz.downloaded_receipt' in files

    data_path = os.path.join(tmpdir, 'version_0')
    assert os.path.exists(data_path)

    # --------------------------
    # test no download happens now, but files are still there
    # --------------------------
    dm = LightningDataModule(tmpdir)
    dm.add_data_url('https://pl-public-data.s3.amazonaws.com/test_0.tar.gz')
    num_urls_downloaded = dm.download_registered_data_urls()

    # we downloaded 0 files
    assert num_urls_downloaded == 0

    files = os.listdir(tmpdir)
    assert '.test_0.tar.gz.downloaded_receipt' in files

    data_path = os.path.join(tmpdir, 'version_0')
    assert os.path.exists(data_path)
