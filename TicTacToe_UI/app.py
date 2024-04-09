from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from model_file import ResNet, TicTacToe

app = Flask(__name__)
CORS(app)

game = TicTacToe()
device = torch.device("cpu")  
model = ResNet(game, 4, 64, device)
model.load_state_dict(torch.load('model_2_TicTacToe.pt', map_location=device))
model.eval()

@app.route('/ai-move', methods=['POST'])
def ai_move():
    data = request.json
    state = np.array(data['state']).reshape((3, 3))
    state_encoded = game.get_encoded_state(state).reshape(1, 3, 3, 3)  
    tensor_state = torch.tensor(state_encoded, dtype=torch.float32, device=device)

    with torch.no_grad():
        policy, _ = model(tensor_state)
        policy = torch.softmax(policy, dim=1).cpu().numpy()
        ai_move = np.argmax(policy)  

    return jsonify({'ai_move': int(ai_move)})

if __name__ == '__main__':
    app.run(debug=True)