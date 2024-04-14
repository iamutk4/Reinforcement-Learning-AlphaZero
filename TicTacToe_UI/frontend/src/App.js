import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const Square = ({ value, onClick }) => (
  <button className="square" onClick={onClick}>
    {value}
  </button>
);

function calculateWinner(squares) {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i];
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
}

const Board = () => {
  const [squares, setSquares] = useState(Array(9).fill(null));
  const [isXTurn, setIsXTurn] = useState(true);

  const handleClick = (i) => {
    if (calculateWinner(squares) || squares[i]) return;
    const newSquares = squares.slice();
    newSquares[i] = isXTurn ? "X" : "O";
    setSquares(newSquares);
    setIsXTurn(!isXTurn);

    if (!isXTurn) return;

    if (
      !calculateWinner(newSquares) &&
      newSquares.every((square) => square !== null)
    ) {
      alert("Draw!");
      return;
    }

    setTimeout(() => {
      const state = newSquares.map((val) =>
        val === "X" ? 1 : val === "O" ? -1 : 0
      );
      axios
        .post("http://127.0.0.1:5000/ai-move", { state })
        .then((response) => {
          const aiMove = response.data.ai_move;
          console.log("AI move:", aiMove);
          if (aiMove !== undefined && newSquares[aiMove] === null) {
            newSquares[aiMove] = "O";
            setSquares(newSquares);
            setIsXTurn(true);

            if (
              !calculateWinner(newSquares) &&
              newSquares.every((square) => square !== null)
            ) {
              alert("Draw!");
            }
          }
        })
        .catch((error) =>
          console.error("There was an error with the AI move:", error)
        );
    }, 200);
  };

  const renderSquare = (i) => (
    <Square value={squares[i]} onClick={() => handleClick(i)} />
  );

  // const winner = calculateWinner(squares);
  // const status = winner ? `Winner: ${winner}` : `Next player: ${isXTurn ? 'X' : 'O'}`;
  const isDraw =
    squares.every((square) => square !== null) && !calculateWinner(squares);
  const status = calculateWinner(squares)
    ? `Winner: ${calculateWinner(squares)}`
    : isDraw
    ? "Draw!"
    : `Next player: ${isXTurn ? "X" : "O"}`;

  return (
    <div>
      <div className="status">{status}</div>
      <div className="board-row">
        {renderSquare(0)}
        {renderSquare(1)}
        {renderSquare(2)}
      </div>
      <div className="board-row">
        {renderSquare(3)}
        {renderSquare(4)}
        {renderSquare(5)}
      </div>
      <div className="board-row">
        {renderSquare(6)}
        {renderSquare(7)}
        {renderSquare(8)}
      </div>
    </div>
  );
};

const App = () => {
  return (
    <div className="App">
      <h1>Tic Tac Toe</h1>
      <Board />
    </div>
  );
};

export default App;
