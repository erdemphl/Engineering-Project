import math
from flask import Flask, render_template, request, redirect, url_for, jsonify
from MCTS import MCTS
from dama.DamaGame import Dama
from dama.pytorch.NNet import NNetWrapper as NNet
from dama.Digits import int2base
import numpy as np
from utils import *

app = Flask(__name__)

g = None
n1 = None
args1 = None
mcts1 = None
ai = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ready', methods=['POST'])
def ready():
    data = request.json
    board_size = data.get('board_size')
    piece_color = data.get('piece_color')
    # Oyun hazır olduğunda yapılacak işlemler burada olacak.
    return '', 204  # No Content, successful

@app.route('/start_game', methods=['GET', 'POST'])
def start_game():
    if request.method == 'POST':
        # Oyun başlatma işlemleri burada yapılacak.
        return '', 204  # No Content, successful
    board_size = request.args.get('board_size')
    piece_color = request.args.get('piece_color')
    return redirect(url_for('board', board_size=board_size, piece_color=piece_color))

@app.route('/board')
def board():
    global g
    global n1
    global args1
    global mcts1
    global ai

    models = {5: 'board5.pth.tar', 6: 'board6.pth.tar', 8: 'board8.pth.tar'}
    board_size = request.args.get('board_size')
    piece_color = request.args.get('piece_color')
    board_size = int(board_size)

    g = Dama(board_size)
    n1 = NNet(g)
    n1.load_checkpoint('./test_models/', models[board_size])
    args1 = dotdict({'numMCTSSims': 400, 'cpuct': 1})
    mcts1 = MCTS(g, n1, args1)
    ai = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    return render_template('board.html', board_size=str(board_size), piece_color=piece_color)

@app.route('/ai_move', methods=['POST'])
def ai_move():
    # Bu fonksiyon yapay zekanın hamlesini hesaplayacak
    global g
    global n1
    global args1
    global mcts1
    global ai
    data = request.json
    board_state = data.get('board_state')
    piece_color = data.get('piece_color')

    # Tahta durumunu numpy matrisi olarak göster
    board = np.array(board_state)
    board = np.rot90(board, 2)
    temp = {"white": 1, "black": -1}
    board = g.getCanonicalForm(board, temp[piece_color[: 5]])
    action = ai(board)
    action = int2base(action, g.n, 4)
    action = getCanonicalMove(action)

    # Geçerli hamlelerin olup olmadığını kontrol et
    valids = g.getValidMoves(board, 1)
    if np.argmax(valids) == 0:
        return jsonify({'message': 'Game is draw.'})

    # Yapay zeka modelini burada çağıracağız ve bir hamle hesaplayacağız
    ai_move = {
        'from': action[:2],
        'to': action[2:]
    }

    print(f"AI Move: {ai_move}")

    return jsonify(ai_move)

@app.route('/user_move', methods=['POST'])
def user_move():
    global g
    global n1
    global args1
    global mcts1
    global ai

    data = request.json
    board_state = data.get('board_state')
    piece_color = data.get('piece_color')
    board = np.array(board_state)
    temp = {"black-piece": -1, "white-piece": 1}
    valids = g.getValidMoves(g.getCanonicalForm(board, temp[piece_color]), 1)

    if np.argmax(valids) == 0:
        return jsonify({'message': 'Game is draw.'})
    else:
        return jsonify({'message': 'User move processed'})

def getCanonicalMove(action):
    global g
    temp = []
    n = g.n - 1
    for index in action:
        temp.append(n - index)
    return temp

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
