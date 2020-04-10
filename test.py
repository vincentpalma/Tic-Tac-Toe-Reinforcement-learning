      if turn == player:                # Si c'est au tour de Q-learning de jouer
        tradeoff = np.random.random()   # Exp vs Exp tradeoff   
        
        if tradeoff < epsilon:          # 1. MOVE Random
          i_St = find_indice(Q,board)
          move = np.random.permutation(possible_actions(board))[0]  
          board[move[0],move[1]] = turn
          i_St1 = find_indice(Q,board)
        else:                           # 2. MOVE Optimal
          i_St = find_indice(Q,board)   # Trouve indice de S(t)
          max_P = -inf                        # Set max_P à -infini
          for a in Q[i_St,1]:           # Pour chaque action dispo au state S(t)
            board[a[0],a[1]] = turn           # On teste la position
            i = find_indice(Q,board)
            if Q[i,2] > max_P:  
              i_St1 = i                       # On garde seulement l'indice de la meilleure valeur P
            board[a[0],a[1]] = 0              # On remet la grille comme avant

          move = Q[i_St1,1]             # On fait le meilleur move
          board[move[0],move[1]] = turn

        if score_eval(board,turn) != 0:       # Calculer max Q(s',a')
          max_Qsa2 = score_eval(board,turn)
        else:
          turn = 1 if turn == 2 else 2
          move = opponent.move(board,turn)
          board[move[0],move[1]] = turn
          
          turn = 1 if turn == 2 else 2
          i_St2 = find_indice(Q,board)   # Trouve indice de S(t)
          max_P = -inf                        # Set max_P à -infini
          for a in Q[i_St2,1]:           # Pour chaque action dispo au state S(t)
            board[a[0],a[1]] = turn           # On teste la position
            print(Q)
            i = find_indice(Q,board)
            if Q[i,2] > max_P:  
              i_St3 = i                       # On garde seulement l'indie de la meilleure valeur P
            board[a[0],a[1]] = 0              # On remet la grille comme avant
          max_Qsa2 = Q[i_St3,2]         # On assigne la meilleure valeur P 

        
        # Actualiser Q selon : Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)] 
        Q[i_St1,2] = Q[i_St1,2] + learning_rate*(score_eval(Q[i_St1,0]) + gamma * max_Qsa2 - Q[i_St1,2])

        turn = 1 if turn == 2 else 2
      else:
        move = opponent.move(board,turn)
        board[move[0],move[1]] = turn
        turn = 1 if turn == 2 else 2


    if (episode+1) % (total_episodes / 10) == 0:
      epsilon = max(0, epsilon - 0.1)






  def move(self,board,turn,Q=None):           # MOVE Optimal
    i_St = find_indice(Q,board)   # Trouve indice de S(t)
    max_P = -inf                  # Set max_P à -infini
    for a in Q[i_St,1]:           # Pour chaque action dispo au state S(t)
      print('a =',a)
      board[a[0],a[1]] = turn     # On teste la position
      
      i = find_indice(Q,board)
      if Q[i,2] > max_P:  
        max_a = a                 # On garde seulement l'indice de la meilleure valeur P

      board[a[0],a[1]] = 0        # On remet la grille comme avant
    print('max_a =',max_a)
    return max_a                  # On return le meilleur move