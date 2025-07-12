import unittest

from src.neuronalNetworks.neuronal_NetworkX import NeuronalNetworkX


#class MyTestCase(unittest.TestCase):
#    def test_something(self):
#       self.assertEqual(True, False)  # add assertion here



if __name__ == '__main__':
    unittest.main()

class FullyConnectedNetworkXTest(unittest.TestCase):
    def test_TopDownTrue(self):
        nxx = NeuronalNetworkX(1,1)
        nxx.defaultNetwork(1,2)
        self.assertEqual(nxx.isFullyConnectedTopDown(), True)

    def test_TopDownFalse(self):
        nnx= NeuronalNetworkX(1,1)
        nnx.defaultNetwork(1,2)
        nnx.remove_edge(2,1)
        self.assertEqual(nnx.isFullyConnectedTopDown(), False)

    def test_BottonUpTrue(self):
        nnx= NeuronalNetworkX(1,1)
        nnx.defaultNetwork(1,2)
        self.assertEqual(nnx.isFullyConnectedBottomUp(), True)

    def test_BottonUpFalse(self):
        nnx= NeuronalNetworkX(1,1)
        nnx.defaultNetwork(1,2)
        nnx.remove_edge(0,2)
        self.assertEqual(nnx.isFullyConnectedBottomUp(), False)