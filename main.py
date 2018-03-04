# -*- coding: UTF-8 -*-
import json

import numpy as np

from gorule import GoRule
from flask import Flask, request, send_file
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
io = SocketIO(app)

board_size = 9
go_rule = GoRule(board_size)

black_boards = [np.zeros((board_size, board_size), dtype=int) for _ in range(8)]
white_boards = [np.zeros((board_size, board_size), dtype=int) for _ in range(8)]
turn = 'black'


def _current_board():
    tmp_board = np.zeros((board_size, board_size), dtype=int)
    for row in range(board_size):
        for col in range(board_size):
            if black_boards[0][row, col] == 1:
                tmp_board[row, col] = 1
            elif white_boards[0][row, col] == 1:
                tmp_board[row, col] = -1
    return tmp_board


def _board_rearrange():
    tmp_array = []
    if turn == 'black':
        for i in range(8):
            tmp_array.append(black_boards[i])
            tmp_array.append(white_boards[i])
        tmp_array.append(np.ones((board_size, board_size), dtype=int))
    else:
        for i in range(8):
            tmp_array.append(white_boards[i])
            tmp_array.append(black_boards[i])
        tmp_array.append(np.zeros((board_size, board_size), dtype=int))
    return np.array(tmp_array).transpose((1, 2, 0))


@app.route('/move', methods=['POST'])
def move():
    global turn

    row = int(request.values.get('row'))
    col = int(request.values.get('col'))

    result = go_rule.move(_board_rearrange(), row * board_size + col)
    if result is not None:
        Xtp1_board, Ytp1_board = result
        black_boards.pop()
        white_boards.pop()
        if turn == 'black':
            black_boards.insert(0, Xtp1_board)
            white_boards.insert(0, Ytp1_board)
            turn = 'white'
        else:
            black_boards.insert(0, Ytp1_board)
            white_boards.insert(0, Xtp1_board)
            turn = 'black'

    return json.dumps(_current_board().tolist())


@app.route('/judge', methods=['POST'])
def judge():
    tmp_board = go_rule.judge(_board_rearrange())
    return json.dumps(tmp_board.tolist())


@app.route('/resign', methods=['POST'])
def resign():
    global black_boards
    global white_boards
    global turn

    black_boards = [np.zeros((board_size, board_size), dtype=int) for _ in range(8)]
    white_boards = [np.zeros((board_size, board_size), dtype=int) for _ in range(8)]
    turn = 'black'
    return json.dumps(_current_board().tolist())


@app.route('/<page_name>.html', methods=['GET'])
def send_html(page_name):
    return send_file(f'{page_name}.html')


@app.route('/test', methods=['POST'])
def test():
    return 'hello world'


@io.on('move2')
def move2(row, col):
    global turn

    row = int(request.values.get('row'))
    col = int(request.values.get('col'))

    result = go_rule.move(_board_rearrange(), row * board_size + col)
    if result is not None:
        Xtp1_board, Ytp1_board = result
        black_boards.pop()
        white_boards.pop()
        if turn == 'black':
            black_boards.insert(0, Xtp1_board)
            white_boards.insert(0, Ytp1_board)
            turn = 'white'
        else:
            black_boards.insert(0, Ytp1_board)
            white_boards.insert(0, Xtp1_board)
            turn = 'black'


if __name__ == '__main__':
    # io.run(app)
    app.run('127.0.0.1', port=5000)