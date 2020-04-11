import numpy as np 

import agents
from main import win_eval

def train(X_player,O_player,episodes):
  ''' Similaire à game(), Q_learning doit être un des deux joueurs '''
  board = np.zeros((3,3),int)   # On crée une matrice 3x3 vide
  turn = 2                      # X est le premier à jouer (règles du jeu)

  while win_eval(board) == 0:   # Tant que la partie n'est pas finie    
    if turn == 2: 
      move = X_player.move(board,turn)
      board[move[0],move[1]] = 2
      turn = 1

    else:
      move = O_player.move(board,turn)
      board[move[0],move[1]] = 1
      turn = 2

  if show_end_state:            # On montre le résultat en image, obligatoire si il y a un joueur humain
    agents.Human().show_end_state(board)  

  return(win_eval(board))       # La fonction renvoie le résultat