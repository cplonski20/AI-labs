import heapq

# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search 
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state)
    #   - keep track of the distance of each state from start
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------

    # ------------------------------
    
    while (len(frontier) > 0):
        currstate = heapq.heappop(frontier)
        if(currstate.is_goal()):
            return backtrack(visited_states, currstate) 
        
        neighbs = currstate.get_neighbors()

        for i in range(len(neighbs)):
            if neighbs[i] in visited_states:
                if neighbs[i].dist_from_start < visited_states[neighbs[i]][1]:
                    visited_states[neighbs[i]] = (currstate, neighbs[i].dist_from_start)
                    # if neighbs[i] in frontier:
                    #     frontier.remove(neighbs[i])
                    heapq.heappush(frontier, neighbs[i])
            else:
                visited_states[neighbs[i]] = (currstate, neighbs[i].dist_from_start) 
                heapq.heappush(frontier, neighbs[i])
            
        

    return []
    # if you do not find the goal return an empty list

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    path = []
    curr = goal_state
    # Your code here ---------------
    while(curr is not None):
        path.append(curr)
        curr = visited_states[curr][0]
    
    path = path[::-1]
    # ------------------------------
    return path