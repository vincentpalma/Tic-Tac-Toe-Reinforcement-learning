import numpy as np 
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt 

import agents
from agents import win_eval

def train_as_X(X_player,O_player,episodes,plot=False):
  ''' Similaire à game(), Q_learning est X '''
  t = tqdm(total=episodes,desc='Training')

  results = []
  for episode in range(episodes):
    board = np.zeros((3,3),int)   # On crée une matrice 3x3 vide

    while win_eval(board) == 0:   # Tant que la partie n'est pas finie    
      ################# X MOVE ######################
      move = X_player.move(board,2)

      S = np.copy(board)     
      A = move

      board[move[0],move[1]] = 2
      
      if win_eval(board) != 0:
        reward = agents.score_eval(board,2)
        prev = X_player.q(S,X_player.format(A))
        X_player.q_table[(X_player.encode(S),X_player.format(A))] = prev + X_player.alpha * (reward + X_player.gamma*reward - prev)
        break

      ################# O MOVE ######################
      move = O_player.move(board,1)
      board[move[0],move[1]] = 1

      S1 = np.copy(board)

      X_player.epsilon = 0    # Pour faire le move optimal (imaginaire)
      A1 = X_player.move(board,2)
      X_player.epsilon = 0.2

      reward = agents.score_eval(board,2)
      X_player.learn(S,A,S1,A1,reward)
    
    results.append([episode,agents.score_eval(board,2)])
    t.update(1)
  t.close()
  
  if plot:
    df = pd.DataFrame(results,columns=['Episode','Result'])
    exp = df.Result.ewm(span=(episodes+9)//10, adjust=False).mean()

    plt.plot(df.Episode,df.Result, 'ro')
    plt.plot(df.Episode,exp, label='EMA')
    plt.xlabel('Episode')
    plt.ylabel('Result (1= X win/-1= O win)')
    plt.show()

  return(X_player.q_table)       # La fonction renvoie Q