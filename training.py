import numpy as np 
import pickle
from tqdm import tqdm 

import agents
from main import win_eval

def train(X_player,O_player,episodes,show_end_state=False):
  ''' Similaire à game(), Q_learning doit être un des deux joueurs '''
  t = tqdm(total=episodes)
  for episode in range(episodes):
    board = np.zeros((3,3),int)   # On crée une matrice 3x3 vide
    turn = 2                      # X est le premier à jouer (règles du jeu)

    while win_eval(board) == 0:      
      if turn == 2: 
        if X_player.player == 'Q' and np.count_nonzero(board) >= 2: # Si on est au moins au move n2 pour apprendre (TD)
          X_player.learn(X_player.board,X_player.last_move,rewardX,board)
        
        move = X_player.move(board,turn)
        board[move[0],move[1]] = 2
        
        if X_player.player == 'Q':
          rewardX = X_player.reward(board,2)
        if O_player.player == 'Q':
          rewardO = O_player.reward(board,1)
        turn = 1

      else:
        if O_player.player == 'Q' and np.count_nonzero(board) >= 3:
          O_player.learn(O_player.board,O_player.last_move,rewardO,board)
        move = O_player.move(board,turn)
        board[move[0],move[1]] = 1

        if X_player.player == 'Q':
          rewardX = X_player.reward(board,2)
        if O_player.player == 'Q':
          rewardO = O_player.reward(board,1)
        turn = 2

    if show_end_state:            # On montre le résultat en image, obligatoire si il y a un joueur humain
      agents.Human().show_end_state(board)  
    t.update(1)
  t.close()

if __name__ == "__main__":
  X_player = agents.Random()
  O_player = agents.Q_learning()
  train(X_player,O_player,50000)

  pickle.dump(O_player.q_table,open('q.QTABLE','wb'))
  # for i in O_player.q_table:
  #   if O_player.q_table[i] !=1:
  #     print(i,O_player.q_table[i])