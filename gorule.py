from __future__ import division

import numpy as np


class GoRule:
    def __init__(self, board_size):
        self.board_size = board_size

    def _clean_dead(self, tmp_board):
        # clean opposite dead
        visit_label = np.zeros((self.board_size, self.board_size), dtype=int)
        search_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for row in range(self.board_size):
            for col in range(self.board_size):
                if visit_label[row, col] == 0 and tmp_board[row, col] == -1:
                    dead_flag = True
                    conn_comp = [(row, col)]
                    index = 0
                    while index < len(conn_comp):
                        (cur_row, cur_col) = conn_comp[index]
                        for (offset_row, offset_col) in search_dirs:
                            tmp_row = cur_row + offset_row
                            tmp_col = cur_col + offset_col
                            if tmp_row < 0 or tmp_row >= self.board_size \
                                    or tmp_col < 0 or tmp_col >= self.board_size \
                                    or visit_label[tmp_row, tmp_col] == 1:
                                pass
                            else:
                                if tmp_board[tmp_row, tmp_col] == 1:
                                    pass
                                elif tmp_board[tmp_row, tmp_col] == -1:
                                    conn_comp.append((tmp_row, tmp_col))
                                    visit_label[tmp_row, tmp_col] = 1
                                else:
                                    dead_flag = False
                        index += 1
                    if dead_flag:
                        for (it_row, it_col) in conn_comp:
                            tmp_board[it_row, it_col] = 0

        # clean own dead
        visit_label = np.zeros((self.board_size, self.board_size), dtype=int)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if visit_label[row, col] == 0 and tmp_board[row, col] == 1:
                    dead_flag = True
                    conn_comp = [(row, col)]
                    index = 0
                    while index < len(conn_comp):
                        (cur_row, cur_col) = conn_comp[index]
                        for (offset_row, offset_col) in search_dirs:
                            tmp_row = cur_row + offset_row
                            tmp_col = cur_col + offset_col
                            if tmp_row < 0 or tmp_row >= self.board_size \
                                    or tmp_col < 0 or tmp_col >= self.board_size \
                                    or visit_label[tmp_row, tmp_col] == 1:
                                pass
                            else:
                                if tmp_board[tmp_row, tmp_col] == 1:
                                    conn_comp.append((tmp_row, tmp_col))
                                    visit_label[tmp_row, tmp_col] = 1
                                elif tmp_board[tmp_row, tmp_col] == -1:
                                    pass
                                else:
                                    dead_flag = False
                        index += 1
                    if dead_flag:
                        for (it_row, it_col) in conn_comp:
                            tmp_board[it_row, it_col] = 0

    def move(self, boards, pos):
        if pos >= self.board_size * self.board_size:  # pass move
            print('pass')
            Ytp1_board = boards[:, :, 1].copy()
            Xtp1_board = boards[:, :, 0].copy()
        else:
            # illegal move
            row = pos // self.board_size
            col = pos % self.board_size
            print(f'{row}, {col}')

            # already has a tone
            if boards[row, col, 0] == 1 or boards[row, col, 1] == 1:
                return None

            # tmp_board represent current state (1 for current player, 0 for empty, -1 for opposite)
            tmp_board = boards[:, :, 0].copy()
            tmp_board -= boards[:, :, 1]
            tmp_board[row, col] = 1
            self._clean_dead(tmp_board)

            Xtp1_board = np.zeros((self.board_size, self.board_size), dtype=int)
            Ytp1_board = np.zeros((self.board_size, self.board_size), dtype=int)
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if tmp_board[row, col] == 1:
                        Xtp1_board[row, col] = 1
                    elif tmp_board[row, col] == -1:
                        Ytp1_board[row, col] = 1

            # judge recurrent state
            for i in range(8):
                if (Xtp1_board == boards[:, :, 2 * i]).all() and (Ytp1_board == boards[:, :, 2 * i + 1]).all():
                    return None

        # legal move return new state
        new_boards = [Ytp1_board, Xtp1_board]
        for i in range(7):
            new_boards.append(boards[:, :, 2 * i + 1])
            new_boards.append(boards[:, :, 2 * i])
        if boards[0, 0, 16] == 1:
            new_boards.append(np.zeros((self.board_size, self.board_size), dtype=int))
        else:
            new_boards.append(np.ones((self.board_size, self.board_size), dtype=int))
        return np.array(new_boards).transpose((1, 2, 0))

    def judge(self, boards):
        # tmp_board represent current state (1 for black, 0 for empty, -1 for white)
        if boards[0, 0, 16] == 1:  # black turn
            tmp_board = boards[:, :, 0].copy()
            tmp_board -= boards[:, :, 1]
        else:  # white turn
            tmp_board = boards[:, :, 1].copy()
            tmp_board -= boards[:, :, 0]

        # find enclosed area
        visit_label = np.zeros((self.board_size, self.board_size), dtype=int)
        search_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for row in range(self.board_size):
            for col in range(self.board_size):
                if visit_label[row, col] == 0 and tmp_board[row, col] == 0:
                    black_flag = False
                    white_flag = False
                    conn_comp = [(row, col)]
                    index = 0
                    while index < len(conn_comp):
                        (cur_row, cur_col) = conn_comp[index]
                        for (offset_row, offset_col) in search_dirs:
                            tmp_row = cur_row + offset_row
                            tmp_col = cur_col + offset_col
                            if tmp_row < 0 or tmp_row >= self.board_size \
                                    or tmp_col < 0 or tmp_col >= self.board_size \
                                    or visit_label[tmp_row, tmp_col] == 1:
                                pass
                            else:
                                if tmp_board[tmp_row, tmp_col] == 1:
                                    black_flag = True
                                elif tmp_board[tmp_row, tmp_col] == -1:
                                    white_flag = True
                                else:
                                    conn_comp.append((tmp_row, tmp_col))
                                    visit_label[tmp_row, tmp_col] = 1
                        index += 1
                    if black_flag and not white_flag:
                        for (it_row, it_col) in conn_comp:
                            tmp_board[it_row, it_col] = 1
                    elif white_flag and not black_flag:
                        for (it_row, it_col) in conn_comp:
                            tmp_board[it_row, it_col] = -1
        return tmp_board
