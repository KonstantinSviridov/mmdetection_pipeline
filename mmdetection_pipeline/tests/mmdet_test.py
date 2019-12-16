import unittest
from musket_core import projects
from musket_core import parralel
import os


fl=__file__
fl=os.path.dirname(fl)

class TestCoders(unittest.TestCase):
    def test_basic_network(self):
        pr = projects.Project(os.path.join(fl, "project"))
        exp = pr.byName("exp01")
        tasks = exp.fit()
        executor = parralel.get_executor(1, 1)
        executor.execute(tasks)
        r = exp.result()
        self.assertGreaterEqual(r, 0, "Result should be greater then zero")
        self.assertTrue(isinstance(r, float), "result should be float")
        print(r)
        pass