from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from model_file import ResNet, TicTacToe, MCTS, Node

app = Flask(__name__)
CORS(app)

game = TicTacToe()
device = torch.device("cpu")  
model = ResNet(game, 4, 64, device='cpu')
model.load_state_dict(torch.load('model_9_TicTacToe.pt', map_location=device))
model.eval()


# MCTS parameters
args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 10,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 0.7,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

# Initialize MCTS
mcts = MCTS(game, args, model)


@app.route('/ai-move', methods=['POST'])
def ai_move():
    data = request.json
    # print(data)
    state = np.array(data['state']).reshape((3, 3))
    # print(state)
    

    player = -1  # Assuming player info is sent, else defaults to player 1

    # Perform MCTS search
    neutral_state = game.change_perspective(state, player)
    mcts_probs = mcts.search(neutral_state)
    ai_move = np.argmax(mcts_probs)

    return jsonify({'ai_move': int(ai_move)})

if __name__ == '__main__':
    app.run(debug=True)



