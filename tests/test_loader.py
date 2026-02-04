"""Unit tests for data loader."""
import unittest
from convqa_eval import get_tasks, list_available_tasks


class TestDataLoader(unittest.TestCase):
    
    def test_list_available_tasks(self):
        tasks = list_available_tasks()
        self.assertIsInstance(tasks, dict)
        self.assertIn("quac", tasks)
    
    def test_get_tasks_single(self):
        tasks = get_tasks(["quac"])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["name"], "quac")
    
    def test_get_tasks_multiple(self):
        tasks = get_tasks(["quac", "coqa"])
        self.assertEqual(len(tasks), 2)
    
    def test_invalid_task(self):
        with self.assertRaises(ValueError):
            get_tasks(["invalid_task"])


if __name__ == "__main__":
    unittest.main()
