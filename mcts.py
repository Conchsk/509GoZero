import numpy as np

from gorule import GoRule
from resnet import ResNet


class TreeNode:
    def __init__(self, boards):
        self.boards = boards.copy()
        board_size = np.shape(self.boards)[0]
        self.N = 0  # visit count
        self.W = 0.0  # total action value
        self.Q = 0.0  # mean action value
        self.P = 0.0  # prior probability
        self.parent = None
        self.children = []


class MCTS:
    def __init__(self, boards):
        self.root = TreeNode(boards)
        self.board_size = np.shape(boards)[0]
        self.go_rule = GoRule(self.board_size)
        self.res_net = ResNet(self.board_size)

    def _puct_calc(self, node: TreeNode):
        return 1

    def _simulate(self):
        # select a leaf node
        it = self.root
        while len(it.children) != 0:
            pos = self._puct_calc(it)
            it = it.children[pos]

        # expand children
        node = it
        policy, value = self.res_net.predict([node.boards])
        for pos in range(self.board_size * self.board_size + 1):
            new_boards = self.go_rule.move(node.boards, pos)
            if new_boards is not None:
                child = TreeNode(new_boards)
                child.P = policy[0, pos]
                child.parent = node
                node.children.append(child)
            else:
                node.children.append(None)

        # backup value
        it = node
        while it is not None:
            it.N += 1
            it.W += value[0, 0]
            it.Q = it.W / it.N
            it = it.parent

    def best_move(self, step):
        it = self.root
        if step < 30:
            pos = np.random.choice(len(it.children), p=it.P / it.P.sum())
        else:
            pos = it.children.index(max(it.children, key=lambda x: x.N))
        self.manual_move(pos)

    def manual_move(self, pos):
        it = self.root
        if it.children[pos] is not None:
            it = it.children[pos]
        return self.root.boards
