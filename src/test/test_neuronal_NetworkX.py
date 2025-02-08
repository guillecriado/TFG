import unittest

from src.neuronalNetworks.neuronal_NetworkX import NeuronalNetworkX


#class MyTestCase(unittest.TestCase):
#    def test_something(self):
#       self.assertEqual(True, False)  # add assertion here



if __name__ == '__main__':
    unittest.main()

class FullyConnectedNetworkXTest(unittest.TestCase):
    def test_TopDownTrue(self):
        nnx= NeuronalNetworkX(1,1)
        nnx.defaultNetwork()
        self.assertEqual(nnx.isFullyConnectedTopDown(), True)
    def test_TopDownFalse(self):
        nnx= NeuronalNetworkX(1,1)
        nnx.defaultNetwork()
        nnx.remove_edge(2,1)
        self.assertEqual(nnx.isFullyConnectedTopDown(), False)