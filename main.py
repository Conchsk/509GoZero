# -*- coding: UTF-8 -*-
import json

import numpy as np

from gorule import GoRule
from mcts import MCTS

from flask import Flask, request, send_file
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
io = SocketIO(app)

board_size = 19
boards = np.zeros((board_size, board_size, 17), dtype=int)
boards[:, :, 16] = np.ones((board_size, board_size))

go_rule = GoRule(board_size)
mcts = MCTS(boards)
global_step = 1


def _current_board():
    if boards[0, 0, 16] == 1:  # black turn
        tmp_board = boards[:, :, 0].copy()
        tmp_board -= boards[:, :, 1]
    else:  # white turn
        tmp_board = boards[:, :, 1].copy()
        tmp_board -= boards[:, :, 0]
    return tmp_board


@app.route('/judge', methods=['POST'])
def judge():
    tmp_board = go_rule.judge(boards)
    return json.dumps(tmp_board.tolist())


@app.route('/resign', methods=['POST'])
def resign():
    global boards
    boards = np.zeros((board_size, board_size, 17), dtype=int)
    boards[:, :, 16] = np.ones((board_size, board_size))
    return json.dumps(_current_board().tolist())


@app.route('/<page_name>.html', methods=['GET'])
def send_html(page_name):
    return send_file(f'{page_name}.html')


@io.on('move1')
def move1(message):
    global boards
    global global_step

    row = message['row']
    col = message['col']

    new_boards = go_rule.move(boards, row * board_size + col)
    if new_boards is not None:
        boards = new_boards
        global_step += 1
        emit('move2', json.dumps(_current_board().tolist()))


@io.on('move2')
def move2(message):
    global boards
    global global_step

    row = message['row']
    col = message['col']

    # player go
    new_boards = go_rule.move(boards, row * board_size + col)
    if new_boards is not None:
        boards = new_boards
        global_step += 1
        emit('move2', json.dumps(_current_board().tolist()))

        # ai go
        mcts.manual_move(row * board_size + col)
        ai_pos = mcts.best_move(global_step)
        new_boards = go_rule.move(boards, ai_pos)
        boards = new_boards
        global_step += 1
        emit('move2', json.dumps(_current_board().tolist()))


if __name__ == '__main__':
    io.run(app, host='127.0.0.1', port=5000)
