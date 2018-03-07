import numpy as np

from gorule import GoRule
from resnet import ResNet


class TreeNode:
    def __init__(self, boards):
        self.boards = boards.copy()
        self.N = np.zeros(1)  # visit count
        self.W = np.zeros(1)  # total action value
        self.Q = np.zeros(1)  # mean action value
        self.P = np.zeros(1)  # prior probability
        self.parent = None
        self.parent_index = -1
        self.children = []


class MCTS:
    def __init__(self, boards):
        self.root = TreeNode(boards)
        self.board_size = np.shape(boards)[0]
        self.go_rule = GoRule(self.board_size)
        self.res_net = ResNet(self.board_size)

    def _puct_calc(self, node: TreeNode, use_dirichlet: bool):
        P_array = node.P / node.P.sum()
        if use_dirichlet:
            P_array += np.random.dirichlet([0.03], np.shape(node.P)[0])
            P_array = P_array / P_array.sum()
        N_array = np.sqrt(node.N.sum()) / (1 + node.N)
        U_array = 0.1 * P_array * N_array
        return U_array

    def _backup(self, node: TreeNode, value: float):
        index = node.parent_index
        it = node.parent
        while it is not None:
            it.N[index] += 1
            it.W[index] += value
            it.Q[index] = it.W[index] / it.N[index]
            index = it.parent_index
            it = it.parent
            value = -value

    def _expand(self, node: TreeNode):
        pos_num = self.board_size * self.board_size + 1

        policy, value = self.res_net.predict([node.boards])
        node.N = np.zeros(pos_num, dtype=int)
        node.W = np.zeros(pos_num, dtype=float)
        node.Q = np.zeros(pos_num, dtype=float)
        node.P = policy[0, :]

        for pos in range(pos_num):
            new_boards = self.go_rule.move(node.boards, pos)
            if new_boards is not None:
                child = TreeNode(new_boards)
                child.parent = node
                child.parent_index = pos
                node.children.append(child)
            else:
                node.children.append(None)

        self._backup(node, value[0, 0])

    def _simulate(self, use_dirichlet: bool):
        pass_num = 0

        # select a leaf node
        it = self.root
        while len(it.children) != 0:
            U_array = self._puct_calc(it, use_dirichlet)
            C_array = it.Q + U_array
            pos = np.argmax(C_array)
            while it.children[pos] is None:
                C_array[pos] = -1
                pos = np.argmax(C_array)
            it = it.children[pos]

            if pos >= self.board_size * self.board_size:  # pass
                pass_num += 1
            else:
                pass_num = 0

            # a terminate state
            if pass_num >= 2:
                if self.go_rule.judge(it.boards).sum() > 3.75:  # black win
                    if it.boards[0, 0, 16] == 1:  # black get a +1 reward
                        self._backup(it, 1)
                    else:  # white get a -1 punish
                        self._backup(it, -1)
                else:  # white win
                    if it.boards[0, 0, 16] == 1:  # black get a -1 punish
                        self._backup(it, -1)
                    else:  # white get a +1 reward
                        self._backup(it, 1)
                return

        # expand children
        self._expand(it)

    def best_move(self, step: int):
        for _ in range(10):
            self._simulate(step > 30)

        it = self.root
        if step < 30:
            pos = np.random.choice(len(it.children), p=it.N / it.N.sum())
        else:
            pos = np.argmax(it.N)[0]
        self.manual_move(pos)
        return pos

    def manual_move(self, pos: int):
        it = self.root
        if len(it.children) == 0:
            self._expand(it)
        if it.children[pos] is not None:
            it = it.children[pos]
            self.root.children = []
            self.root = it
