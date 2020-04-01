# Tic Tac Toe Reinforcement learning

## Setup
Install requirements with pip

```bash
pip install -r requirements.txt
```
## Run
```bash
python main.py
```
### Change agents
Example :
```python
if __name__ == "__main__":
  X_player = agents.Random()
  O_player = agents.Minimax()
  print(game(X_player,O_player,True))
```