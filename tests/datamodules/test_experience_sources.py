from unittest import TestCase

from torch.utils.data import DataLoader

from pl_bolts.datamodules.experience_source import ExperienceSourceDataset


class TestExperienceSourceDataset(TestCase):
    def train_batch(self):
        """Returns an iterator used for testing"""
        return iter([i for i in range(100)])

    def test_iterator(self):
        """Tests that the iterator returns batches correctly"""
        source = ExperienceSourceDataset(self.train_batch)
        batch_size = 10
        data_loader = DataLoader(source, batch_size=batch_size)

        for idx, batch in enumerate(data_loader):
            self.assertEqual(len(batch), batch_size)
            self.assertEqual(batch[0], 0)
            self.assertEqual(batch[5], 5)
            break
