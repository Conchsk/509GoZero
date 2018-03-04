import numpy as np
import pandas


class TreeNode:
    def __init__(self):
        self.N = np.zeros((19, 19))  # visit count
        self.W = np.zeros((19, 19))  # total action value
        self.Q = np.zeros((19, 19))  # mean action value
        self.P = np.zeros((19, 19))  # prior probability
        self.parent = None
        self.children = []

class MCTS:
    def __init__(self):
        self.root = TreeNode()

    def __