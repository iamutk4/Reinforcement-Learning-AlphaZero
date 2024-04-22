import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const Square = ({ value, onClick }) => (
  <button className="square" onClick={onClick}>
    {value}
  </button>
);

function calculateWinner(squares) {
  // Winning lines for Connect Four
  const lines = [
    // Horizontal
    [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6],
    [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12], [10, 11, 12, 13],
    [14, 15, 16, 17], [15, 16, 17, 18], [16, 17, 18, 19], [17, 18, 19, 20],
    [21, 22, 23, 24], [22, 23, 24, 25], [23, 24, 25, 26], [24, 25, 26, 27],
    [28, 29, 30, 31], [29, 30, 31, 32], [30, 31, 32, 33], [31, 32, 33, 34],
    [35, 36, 37, 38], [36, 37, 38, 39], [37, 38, 39, 40], [38, 39, 40, 41],
    // Vertical
    [0, 7, 14, 21], [1, 8, 15, 22], [2, 9, 16, 23], [3, 10, 17, 24],
    [4, 11, 18, 25], [5, 12, 19, 26], [6, 13, 20, 27],
    [7, 14, 21, 28], [8, 15, 22, 29], [9, 16, 23, 30], [10, 17, 24, 31],
    [11, 18, 25, 32], [12, 19, 26, 33], [13, 20, 27, 34],
    [14, 21, 28, 35], [15, 22, 29, 36], [16, 23, 30, 37], [17, 24, 31, 38],
    [18, 25, 32, 39], [19, 26, 33, 40], [20, 27, 34, 41],
    // Diagonal
    [0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27],
    [7, 15, 23, 31], [8, 16, 24, 32], [9, 17, 25, 33], [10, 18, 26, 34],
    [14, 22, 30, 38], [15, 23, 31, 39], [16, 24, 32, 40], [17, 25, 33, 41],
    [3, 9, 15, 21], [4, 10, 16, 22], [5, 11, 17, 23], [6, 12, 18, 24],
    [10, 16, 22, 28], [11, 17, 23, 29], [12, 18, 24, 30], [13, 19, 25, 31],
  ];

  for (let i = 0; i < lines.length; i++) {
    const [a, b, c, d] = lines[i];
    if (
      squares[a] &&
      squares[a] === squares[b] &&
      squares[a] === squares[c] &&
      squares[a] === squares[d]
    ) {
      return squares[a];
    }
  }
  return null;
}

const Board = () => {
  const [squares, setSquares] = useState(Array(42).fill(null)); // Change array size to 42
  const [isXTurn, setIsXTurn] = useState(true);

  const handleClick = (col) => {
    if (calculateWinner(squares)) return;
    const newSquares = [...squares];
    for (let row = 5; row >= 0; row--) {
      if (!newSquares[row * 7 + col]) {
        newSquares[row * 7 + col] = isXTurn ? 'X' : 'O';
        break;
      }
    }
    setSquares(newSquares);
    setIsXTurn(!isXTurn);

    if (!isXTurn) return;

    if (!calculateWinner(newSquares) && newSquares.every((square) => square !== null)) {
      alert('Draw!');
      return;
    }

    setTimeout(() => {
      const state = newSquares.map((val) => (val === 'X' ? 1 : val === 'O' ? -1 : 0));
      axios
        .post('http://127.0.0.1:5000/ai-move', { state })
        .then((response) => {
          const aiMove = response.data.ai_move;
          const row = response.data.row;
          if (aiMove !== undefined && newSquares[aiMove] === null) {
            const i = row * 7 + aiMove;
            console.log("AI move:", aiMove);
            newSquares[i] = 'O';
            setSquares(newSquares);
            setIsXTurn(true);

            if (!calculateWinner(newSquares) && newSquares.every((square) => square !== null)) {
              alert('Draw!');
            }
          }
        })
        .catch((error) => console.error('There was an error with the AI move:', error));
    }, 200);
  };
  
  const renderSquare = (i) => <Square value={squares[i]} onClick={() => handleClick(i % 7)} />;

  const isDraw = squares.every((square) => square !== null) && !calculateWinner(squares);
  const status = calculateWinner(squares)
    ? `Winner: ${calculateWinner(squares)}`
    : isDraw
    ? 'Draw!'
    : `Next player: ${isXTurn ? 'X' : 'O'}`;

  return (
    <div>
      <div className="status">{status}</div>
      {/* Render the board dynamically */}
      {Array(6)
        .fill()
        .map((_, row) => (
          <div className="board-row" key={row}>
            {Array(7)
              .fill()
              .map((_, col) => renderSquare(row * 7 + col))}
          </div>
        ))}
    </div>
  );
};

const App = () => {
  return (
    <div className="App">
      <h1>Connect Four</h1>
      <Board />
    </div>
  );
};

export default App;
