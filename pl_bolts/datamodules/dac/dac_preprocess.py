"""
Preprocess Criteo dataset. This dataset was used for the Display Advertising
Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
"""
import os
import random
import collections
from tqdm import tqdm

# There are 13 integer features and 26 categorical features
continuous_features = range(1, 14)
categorical_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continuous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorical_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorical_features[i]] != '':
                        self.dicts[i][features[categorical_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature

    def build(self, datafile, continuous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continuous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continuous_clip[i]:
                            val = continuous_clip[i]

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return val


def preprocess_dac(datadir, outdir, num_train_sample=-1, num_test_sample=-1):
    """
    All the 13 integer features are normalzied to continuous values and these
    continuous features are combined into one vecotr with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    dists = ContinuousFeatureGenerator(len(continuous_features))
    dists.build(os.path.join(datadir, 'train.txt'), continuous_features)

    dicts = CategoryDictGenerator(len(categorical_features))
    dicts.build(
        os.path.join(datadir, 'train.txt'), categorical_features, cutoff=200)

    dict_sizes = dicts.dicts_sizes()
    with open(os.path.join(outdir, 'feature_sizes.txt'), 'w') as feature_sizes:
        sizes = [1] * len(continuous_features) + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))

    random.seed(0)

    # Saving the data used for training.
    with open(os.path.join(outdir, 'train_processed.txt'), 'w') as out_train:
        with open(os.path.join(datadir, 'train.txt'), 'r') as f:
            print('processing train features')
            feats = f.readlines()
            feats = feats[:num_train_sample] if num_train_sample >= 0 else feats
            for line in tqdm(feats):
                features = line.rstrip('\n').split('\t')

                continuous_vals = []
                for i, feat in enumerate(continuous_features):
                    val = dists.gen(i, features[feat])
                    continuous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                categorial_vals = []
                for i, feat in enumerate(categorical_features):
                    val = dicts.gen(i, features[feat])
                    categorial_vals.append(str(val))

                continuous_vals = ','.join(continuous_vals)
                categorial_vals = ','.join(categorial_vals)
                label = features[0]
                out_train.write(','.join([continuous_vals, categorial_vals, label]) + '\n')

    with open(os.path.join(outdir, 'test_processed.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            print('processing test features')
            feats = f.readlines()
            feats = feats[:num_test_sample] if num_test_sample >= 0 else feats
            for line in tqdm(feats):
                features = line.rstrip('\n').split('\t')

                continuous_vals = []
                for i, feat in enumerate(continuous_features):
                    val = dists.gen(i, features[feat - 1])
                    continuous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))
                categorial_vals = []
                for i, feat in enumerate(categorical_features):
                    val = dicts.gen(i, features[feat - 1])
                    categorial_vals.append(str(val))

                continuous_vals = ','.join(continuous_vals)
                categorial_vals = ','.join(categorial_vals)
                out.write(','.join([continuous_vals, categorial_vals]) + '\n')
