import numpy as np
import agents

def win_eval(board):
  ''' Fonction qui évalue une grille et renvoie : 0 si la partie n'est pas finie, 1 si O gagne, 2 si X gagne ou 3 si match nul '''
  for x in range(3):
    if board[x,0] == board[x,1] == board[x,2] != 0: return board[x,0]    # -
    elif board[0,x] == board[1,x] == board[2,x] != 0: return board[0,x]  # |
    elif board[0,0] == board[1,1] == board[2,2] != 0: return board[0,0]  # \
    elif board[0,2] == board[1,1] == board[2,0] != 0: return board[0,2]  # /
  if np.count_nonzero(board) < 9: return 0  # Pas fini
  return 3                                  # Match nul

def game(X_player,O_player,show_end_state=True):
    ''' La fonction principale, renvoie le résultat d'une partie entre 2 joueurs '''
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

if __name__ == "__main__":
  X_player = agents.Q_learning()
  O_player = agents.Human()
  print(game(X_player,O_player,True))