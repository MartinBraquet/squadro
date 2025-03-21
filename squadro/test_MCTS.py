from squadro_state import SquadroState
import numpy as np

agent_0 = 'contest_agent3'
agent_1 = 'contest_agent3'

def main():

    cur_state = SquadroState()
    agents = [getattr(__import__(agent_0), 'MyAgent')(), getattr(__import__(agent_1), 'MyAgent')()]
    agents[0].set_id(0)
    agents[1].set_id(1)
    
    cur_state.cur_pos[0][0] = (100, 400)
    cur_state.cur_pos[0][1] = (200, 100)
    cur_state.returning[0][1] = True
    cur_state.finished[0][2] = True
    cur_state.finished[0][3] = True
    cur_state.cur_player = 0
    
    l1 = [cur_state.get_pawn_advancement(cur_state.cur_player, pawn) for pawn in [0, 1, 2, 3, 4]]
    l2 = [cur_state.get_pawn_advancement(1 - cur_state.cur_player, pawn) for pawn in [0, 1, 2, 3, 4]]

    #print(l1, l2)

    for i in range(2,50):
        agents[0].MC_steps = i
        action = agents[0].get_action(cur_state.copy(), 0, 10)
    
    cur_state.apply_action(action)

    #l1 = [cur_state.get_pawn_advancement(0, pawn) for pawn in [0, 1, 2, 3, 4]]
    #l2 = [cur_state.get_pawn_advancement(1, pawn) for pawn in [0, 1, 2, 3, 4]]

    #print(l1, l2)


if __name__ == "__main__":

	main()
