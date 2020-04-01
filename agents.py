from math import inf
import numpy as np
import time

def win_eval(board):      # Je réécris la fonction pour éviter une import loop
  ''' Fonction qui évalue une grille et renvoie : 0 si la partie n'est pas finie, 1 si O gagne, 2 si X gagne ou 3 si match nul '''
  for x in range(3):
    if board[x,0] == board[x,1] == board[x,2] != 0: return board[x,0]    # -
    elif board[0,x] == board[1,x] == board[2,x] != 0: return board[0,x]  # |
    elif board[0,0] == board[1,1] == board[2,2] != 0: return board[0,0]  # \
    elif board[0,2] == board[1,1] == board[2,0] != 0: return board[0,2]  # /
  if np.count_nonzero(board) < 9: return 0  # Pas fini
  return 3                                  # Match nul

class Human:
  ''' input = current state => update GUI, output = mouse click => move + update GUI'''
  def __init__(self):
    import gui           # On importe gui ici seulement car on a pas besoin de pygame si les IAs jouent entre eux
    self.gui = gui.gui() # On initialise le gui ici car move() est la 1ère methode à être appellée

  def move(self,board,turn):       # Renvoie sous forme d'un tuple (row,col) le coup du joueur
    self.board = board
    self.update()

    move = self.decision()

    self.board[move[0],move[1]] = turn
    self.update()
    return move

  def update(self):     # Actualise l'écran
    self.gui.draw_board()
    self.gui.draw_xo(self.board)

  def decision(self):   # Renvoie les la case que l'humain choisi
    return self.gui.play()

  def show_end_state(self,board): # Affiche l'état final avec une ligne si quelqu'un a gagné
    self.board = board
    self.update()
    self.gui.draw_line(board)
    time.sleep(2)


class Random:
  ''' input = board, output = random move'''

  def move(self,board,turn):
    self.board = board
    possible_moves = np.argwhere(self.board == 0)   # On fait une liste de tous les indices avec une valeur de 0
    move = np.random.permutation(possible_moves)[0] # On permute aléatoirement l'array et on prends le premier indice

    return (move[0],move[1])                    # On le retourne sous la forme d'un tuple

############## ALGORITHMES ##############

def win_eval(board):      # Je réécris la fonction pour éviter une import loop
  ''' Fonction qui évalue une grille et renvoie : 0 si la partie n'est pas finie, 1 si O gagne, 2 si X gagne ou 3 si match nul '''
  
  board = np.array(board) if type(board) == list else board   # Si la grille venait a être une liste (minimax utilise des listes)

  for x in range(3):
    if board[x,0] == board[x,1] == board[x,2] != 0: return board[x,0]    # -
    elif board[0,x] == board[1,x] == board[2,x] != 0: return board[0,x]  # |
    elif board[0,0] == board[1,1] == board[2,2] != 0: return board[0,0]  # \
    elif board[0,2] == board[1,1] == board[2,0] != 0: return board[0,2]  # /
  if np.count_nonzero(board) < 9: return 0  # Pas fini
  return 3                                  # Match nul


def score_eval(board,turn):
  ''' Calcule le score à donner à l'algorithme '''
  antiturn = 0
  if turn == 1: 
    antiturn = 2
  elif turn == 2: 
    antiturn = 1
    
  if win_eval(board) == antiturn:   # Si l'autre gagne
    score = -1
  elif win_eval(board) == turn: # Si minimax gagne
    score = 1
  else:
    score = 0
  return score

def minimax_algorithm(board,depth,turn,maximizingPlayer=True):
  ''' Source : https://fr.wikipedia.org/wiki/Algorithme_minimax
      Cet algorithme va explorer l'arbre de tous les coups possible '''
  board = list(board)   # Pour des raisons que j'ignore, ça ne marche pas avec des arrays numpy

  if win_eval(board) != 0:     # Si on se trouve dans un état final
    return score_eval(board,turn)
  if maximizingPlayer:
    best_score = -inf
    for row in range(3):
      for col in range(3):
        if board[row][col] == 0:   # Si la case est vide
          board[row][col] = 1      # Comme minimac est le maximizingPlayer et qu'il joue O, on essaye pour chaque case vide de mettre un O 
          score = minimax_algorithm(board,depth-1,turn,False)     # On calcul le score récursivement
          board[row][col] = 0      # On remet la valeur initiale (ça nous évite de devoir deep copy la liste)
          best_score = max(best_score,score)        # On garde le score maximal
    return best_score
  else:  # On fait exactement la même chose mais avec min(), on suppose que l'adversaire va toujours jouer le meilleur coup possible
    best_score = +inf
    for row in range(3):
      for col in range(3):
        if board[row][col] == 0:
          board[row][col] = 2
          score = minimax_algorithm(board,depth-1,turn,True)
          board[row][col] = 0
          best_score = min(best_score,score)
    return best_score    

def best_move(board,turn):
  ''' Fonction qui va déterminer le meilleur coup à jouer pour minimax selon la grille '''
  best_score = -inf
  for row in range(3):
    for col in range(3):
      if board[row,col] == 0:
        board[row,col] = turn

        score = minimax_algorithm(board,list(np.ravel(board)).count(0),turn,False)   # Pour avoir la depth, on compte le nombre de 0 dans la grille
        
        board[row,col] = 0
        best_score = max(best_score,score)
        if best_score == score:
          move = (row,col)
  return move

class Minimax:
  ''' input = current state, output = new move ''' 
  def move(self,board,turn):
    return best_move(board,turn)


class Q_learning:
  ''' input = current state, output = new move '''
  def __init__(self,board,q_table):
    self.board = board
  pass    # À compléter
