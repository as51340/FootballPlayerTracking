from typing import List

class Player:
    id: int  # matches the id of the identification
    
    def __init__(self, id: int):
        self.id = id


class Referee:
    id: int  # matches the id of the identification
    
    def __init__(self, id: int):
        self.id = id
        
    

class Team:
    color: str
    players: List[Player]
    
    def __init__(self, color: str):
        self.color = color

    def reset(self):
        self.players = []
        
    def add_player(self, player: Player):
        self.players.append(player)
                
        
class Match:
    team1: Team
    team2: Team
    
    def __init__(self):
        pass
    
    def reset(self):
        self.team1.reset()
        self.team2.reset()