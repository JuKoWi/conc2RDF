import sys
sys.path.append("../src")
from conc2RDF import merge_dictionaries
import unittest



class TestMergeDictionaries(unittest.TestCase):

    def test_basic_merge(self):
        default_dict = {'a': 1, 'b': 2}
        custom_dict = {'b': 3, 'c': 4}
        result = merge_dictionaries(default_dict, custom_dict)
        expected = {'a': 1, 'b': 3, 'c': 4}
        self.assertEqual(result, expected)

    def test_nested_merge(self):
        default_dict = {'a': {'x': 1, 'y': 2}, 'b': 2}
        custom_dict = {'a': {'y': 3, 'z': 4}}
        result = merge_dictionaries(default_dict, custom_dict)
        expected = {'a': {'x': 1, 'y': 3, 'z': 4}, 'b': 2}
        self.assertEqual(result, expected)

    def test_only_default_dict(self):
        default_dict = {'a': 1, 'b': {'x': 10}}
        custom_dict = {}
        result = merge_dictionaries(default_dict, custom_dict)
        expected = {'a': 1, 'b': {'x': 10}}
        self.assertEqual(result, expected)

    def test_only_custom_dict(self):
        default_dict = {}
        custom_dict = {'a': 5, 'b': {'y': 20}}
        result = merge_dictionaries(default_dict, custom_dict)
        expected = {'a': 5, 'b': {'y': 20}}
        self.assertEqual(result, expected)

    def test_custom_overrides_non_dict_default(self):
        default_dict = {'a': 1, 'b': {'x': 10}}
        custom_dict = {'b': 20}
        result = merge_dictionaries(default_dict, custom_dict)
        expected = {'a': 1, 'b': 20}
        self.assertEqual(result, expected)

    def test_empty_dicts(self):
        default_dict = {}
        custom_dict = {}
        result = merge_dictionaries(default_dict, custom_dict)
        expected = {}
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
    