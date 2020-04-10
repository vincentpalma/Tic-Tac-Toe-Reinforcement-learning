import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm     # Barre de chargement
import pickle
from itertools import permutations
from math import inf

from agents import score_eval
import agents

def possible_actions(board):
  return np.argwhere(board == 0)

def combinaisons():
  ''' Calcule toutes les positions possibles du jeu (states)
  avec le nombre de X max = 5 et le nombre de O max = 4
  (Par les règles du jeux X joue toujours en permier) 

  PS : Cela inclue aussi les états de jeux ou l'algortithme continue de jouer après que l'autre aie gagné
  bien que ces états n'arriverons jamais en pratique. Selon cette source: http://www.mathrec.org/old/2002jan/solutions.html, 
  exclure ces positions diminuerait le nombre d'états de 6046 à 5478, mais ça serait compliquer le code pour un gain minime en mémoire.
  ''' 
  tours = []
  t = tqdm(total=9,desc="Calculating all possible states",leave=False)
  for x in range(9):    # Pour chaque tour
    if x % 2 == 0:
      tour = []
      for nombre in range(int((x/2)+1)):
        tour.append(2)
      for nombre in range(int(x/2)):
        tour.append(1)
      while len(tour) < 9:
        tour.append(0)
      tours.append(tour)
    else:
      tour = []
      for nombre in range(int((x+1)/2)):
        tour.extend([2,1])
      while len(tour) < 9:
        tour.append(0)
      tours.append(tour)
    t.update(1)
  t.close()

  C = [(0,0,0,0,0,0,0,0,0)]
  t = tqdm(total=9,desc="Deleting duplicate states",leave=False)
  for x in range(9):
    l = list(sorted(set(list(permutations(tours[x]))))) # Trier et supprimer les doublons
    for i in l:
      C.append(i)
    t.update(1)
  t.close()

  t = tqdm(total=len(C),desc="Converting to Numpy array",leave=False)
  for state in range(len(C)): # Conversion au format array de numpy
    C[state] = np.array([[C[state][0],C[state][1],C[state][2]],
                         [C[state][3],C[state][4],C[state][5]],
                         [C[state][6],C[state][7],C[state][8]]])
    t.update(1)
  t.close()

  n = len(C)     # Nombre d'éléments dans C actualisés
  j = 0
  t = tqdm(total=n,desc="Deleting impossibles states",leave=False)
  while j < n:

    i = 0
    for x in range(3):        # Supression des states impossibles (3X et 3O alignés)
      if C[j][x,0] == C[j][x,1] == C[j][x,2] != 0: i+=1    # -
      elif C[j][0,x] == C[j][1,x] == C[j][2,x] != 0: i+=1  # |
      elif C[j][0,0] == C[j][1,1] == C[j][2,2] != 0: i+=1  # \
      elif C[j][0,2] == C[j][1,1] == C[j][2,0] != 0: i+=1  # /
    if i>1:   
      del C[j]
    n = len(C)
    j+=1
    t.total = n
    t.refresh()
    t.update(1)
  t.close()

  C = np.array(C)
  return C

def init_qtable(C,turn):
  ''' Initialise une Qtable avec P de gagner de chaque état = 0.5 sauf aux états terminaux,
      où P=1 ou P=0 selon si X ou O gagne et selon si l'agent joue en tant que X ou O (variable turn) 
      Renvoie un array sous la forme :
      {array( [[State_n],array([[move1_from_Sn],[move2_from_Sn],[...]])] ) : P of winning from Sn, ...}'''
  
  q_table = []
  t = tqdm(total=len(C),desc="Initializing Q-Table",leave=False)
  for state in C:
    if score_eval(state,turn) == 1:
      P = 1
    elif score_eval(state,turn) ==-1:
      P = 0
    else: 
      P = 0.5
    
    q_table.append(np.array([state,possible_actions(state),P]))
    t.update(1)
  t.close()

  return np.array(q_table)

def find_indice(q_table,state):
  ''' Trouve l'indice d'un state (valide) dans la q_table '''
  for i in range(len(q_table)):
    if str(q_table[i,0]) == str(state):
      return i


def train(player,opponent,total_episodes,verbose=True,learning_rate=0.3,gamma=0.9,epsilon=0.2):
  C = combinaisons()
  Q = init_qtable(C,player)   # /!\ Player doit être 1 ou 2

  t = tqdm(total=total_episodes)
  for episode in range(total_episodes):
    board = np.zeros((3,3),int)
    
    turn = 2

    while score_eval(board,turn) == 0:       # Tant que la partie n'est pas finie
      if turn == player:                # Si c'est au tour de Q-learning de jouer
        tradeoff = np.random.random()   # Exp vs Exp tradeoff   
        
        if verbose: print('------------------\n',tradeoff,board)

        i_St = find_indice(Q,board)
        
        if tradeoff < epsilon:
          move = np.random.permutation(possible_actions(board))[0]
          board[move[0],move[1]] = turn
          if verbose: print('moved rand')
        else:
          move =  agents.Q_learning().move(board,turn,Q=Q) 
          board[move[0],move[1]] = turn
          if verbose: print('moved opt')

        i_St1 = find_indice(Q,board)

        if verbose: print(board,"i_St1 =",i_St1,"\n")
        
        # Actualiser Q[P] selon : V(s) := V(s) + lr*[V(s') - V(s)]
        if verbose: 
          print(Q[i_St1])
          print("V(s) := V(s) + lr*[V(s') - V(s)] ===>",Q[i_St,2], "--> ...")
          print(Q[i_St,2], '=', Q[i_St,2], '+', learning_rate,'*','(',Q[i_St1,2],'-',Q[i_St,2],')')
        Q[i_St,2] = Q[i_St,2] + learning_rate*(Q[i_St1,2]-Q[i_St,2])
        if verbose: print(Q[i_St,2])

        turn = 1 if turn == 2 else 2
      else:
        move = opponent.move(board,turn)
        board[move[0],move[1]] = turn
        turn = 1 if turn == 2 else 2
    print(board,score_eval(board,player),'------------------\n')

    if (episode+1) % (total_episodes / 10) == 0:
      epsilon = max(0, epsilon - 0.1)

    t.update(1)
  return Q
  t.close()

new_Q = train(2,agents.Random(),20)
for i in range(len(new_Q)):
  if new_Q[i,2] != 0 and new_Q[i,2] != 1 and new_Q[i,2] != 0.5: print(new_Q[i,0],new_Q[i,2])