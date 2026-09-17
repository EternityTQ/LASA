"""Test actual partition code with synthetic dataset and stdlib serialization.

AST loading excludes torchvision/dill imports only; load_partition and iid
execute unchanged. No dataset download, model fitting or real cache is used.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

SOURCE = Path(__file__).parents[1] / 'utils/data_pre_process.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
FUNCTIONS = ast.Module(body=[node for node in TREE.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name in ('load_partition', 'iid')], type_ignores=[])


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = os.getcwd()
        os.chdir(self.temp.name)
        Path('data/dataset/fmnist').mkdir(parents=True)
        noop = lambda *a, **kw: None
        namespace = dict(np=np, torch=torch, os=os, dill=pickle, json=json, hashlib=hashlib,
                         datasets=SimpleNamespace(FashionMNIST=lambda *a, **kw: list(range(600))),
                         transforms=SimpleNamespace(Compose=noop, ToTensor=noop, Normalize=noop))
        exec(compile(FUNCTIONS, str(SOURCE), 'exec'), namespace)
        self.load = namespace['load_partition']
        self.args = SimpleNamespace(dataset='fmnist', iid=1, seed=1, num_users=100, freeze_datasplit=1)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()

    def test_legacy_cache_ignored_and_rng_identical_on_hit_and_miss(self):
        with open('data/dataset/fmnist/fmnist_dict_users.pik', 'wb') as stream:
            pickle.dump({i: {0} for i in range(6000)}, stream)
        np.random.seed(17)
        initial = np.random.get_state()
        first = self.load(self.args)[-1]
        after_miss = np.random.random(5)
        np.random.set_state(initial)
        second = self.load(self.args)[-1]
        after_hit = np.random.random(5)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(after_miss, after_hit)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set.union(*first.values())), 600)
        self.args.seed = 2
        self.assertNotEqual(first, self.load(self.args)[-1])

    def test_wrong_client_count_and_duplicate_indices_are_rejected(self):
        self.load(self.args)
        path = next(Path('data/dataset/fmnist').glob('fmnist_v2*.pik'))
        for bad in ({0: {1}}, {i: {0, 1, 2, 3, 4, 5} for i in range(100)}):
            with path.open('wb') as stream:
                pickle.dump(bad, stream)
            with self.assertRaises(ValueError):
                self.load(self.args)
        # --freeze_datasplit=0 does not read a corrupt cache.
        self.args.freeze_datasplit = 0
        self.assertEqual(len(self.load(self.args)[-1]), 100)


if __name__ == '__main__':
    unittest.main()
