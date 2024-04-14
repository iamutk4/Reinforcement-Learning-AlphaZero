from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from model_file import ResNet, MCTS, Node, ConnectFour

app = Flask(__name__)
CORS(app)

game = ConnectFour()
device = torch.device("cpu")  
model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load('./model_7_ConnectFour_OSC.pt', map_location=device))
model.eval()


# MCTS parameters
args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}

# Initialize MCTS
mcts = MCTS(game, args, model)


@app.route('/ai-move', methods=['POST'])
def ai_move():
    data = request.json
    state = np.array(data['state']).reshape((6, 7)).astype(np.int32)  # Assuming state is sent as a 6x7 array
    state_encoded = game.get_encoded_state(state).reshape(1, 6, 7, 3)  
    tensor_state = torch.tensor(state_encoded, dtype=torch.float32, device=device)

    player = data.get('player', 1)  # Assuming player info is sent, else defaults to player 1

    # Perform MCTS search
    neutral_state = game.change_perspective(state, player)
    mcts_probs = mcts.search(neutral_state)
    ai_move = np.argmax(mcts_probs)

    print("AI Move Before Placing:", ai_move)
    # Find the lowest empty row in the selected column
    for row in range(5,-1,-1):
        if state[row][ai_move] == 0:
            state[row][ai_move] = player
            print("AI Move Placed at Row:", row)
            print(state)
            break
    return jsonify({'ai_move': int(ai_move), 'row': int(row), 'state': state.astype(np.int32).flatten().tolist()})

if __name__ == '__main__':
    app.run(debug=True)



