import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)
        #np.zeros((NUM_FOOD_DIR_X, NUM_FOOD_DIR_Y, NUM_ADJOINING_WALL_X_STATES, NUM_ADJOINING_WALL_Y_STATES, 
		#			 NUM_ADJOINING_BODY_TOP_STATES, NUM_ADJOINING_BODY_BOTTOM_STATES, NUM_ADJOINING_BODY_LEFT_STATES,
		#			 NUM_ADJOINING_BODY_RIGHT_STATES, NUM_ACTIONS))

        # TODO: write your function here
        # actdict = { 0: utils.UP, 1 : utils.DOWN, 2 : utils.LEFT, 3 : utils.RIGHT}
                # in testing mode
        #edge case at start

        # in testing mode
        if(self._train == False):
            self.s = s_prime
            action = self.actions[np.argmax(self.Q[s_prime]).astype('int')] # THIS COULD BE WRONG BUT NOT A PROBLEM RIGHT NOW
            self.a = action
            self.points = points
            return self.a
        else:
            if(self.s is None or self.a is None):
                actList = np.array([i for i in self.Q[s_prime]])
                Nlist = self.N[s_prime]
                for i in range(Nlist.size):
                    if Nlist[i]< self.Ne:
                        actList[i] = 1
                
                maxidx = 3
                maxVal = actList[3]
                for i in reversed(range(3)):
                    if(actList[i] > maxVal):
                        maxVal = actList[i]
                        maxidx = i

                #update actions points state
                
                self.a = self.actions[maxidx]
                self.s = s_prime
                self.points = points

                return self.a
            
            #death edge case
            if(dead):

                self.N[self.s][self.a] = self.N[self.s][self.a] + 1 #update N
                oldQ = self.Q[self.s][self.a] #get old Q
                currmaxQ = np.amax(self.Q[s_prime])
                alpha = self.C/(self.C + self.N[self.s][self.a])

                #dead so we know reward -1 
                reward = -1

                #UPDATE Q 
                self.Q[self.s][self.a] = oldQ + alpha * (reward + self.gamma*currmaxQ - oldQ)
                self.reset()
                return 0 #arbitrary

            # note in testing mode and not at start
            
            self.N[self.s][self.a] = self.N[self.s][self.a] + 1 #update N
            oldQ = self.Q[self.s][self.a] #get old Q
            currmaxQ = np.amax(self.Q[s_prime]) # get currQ
            alpha = self.C/(self.C + self.N[self.s][self.a])
            #cant be dead so reward either 1 or -0.1
            reward = 0
            if(points > self.points):
                reward = 1

            #UPDATE Q 
            self.Q[self.s][self.a] = oldQ + alpha * (reward + self.gamma*currmaxQ - oldQ)
            
            #Choosing next action
            # note for tie RIGHT > LEFT > DOWN > UP.
            actList = np.array([i for i in self.Q[s_prime]])
            Nlist = self.N[s_prime]
            for i in range(Nlist.size):
                if Nlist[i]< self.Ne:
                    actList[i] = 1
            
            maxidx = 3
            maxVal = actList[3]
            for i in reversed(range(3)):
                if(actList[i] > maxVal):
                    maxVal = actList[i]
                    maxidx = i

            #update actions points state
            self.a = self.actions[maxidx]
            self.s = s_prime
            self.points = points
            return self.a

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        fdirx, fdiry, wx, wy, bt,bl,br,bb = 0,0,0,0,0,0,0,0
        height = utils.DISPLAY_HEIGHT - 2
        width = utils.DISPLAY_WIDTH - 2
        sx,sy,sbod,fx,fy = environment[0], environment[1], environment[2], environment[3], environment[4]
        if(sx == fx):
            fdirx = 0
        elif(sx > fx):
            fdirx = 1
        else:
            fdirx = 2
        
        if sy == fy:
            fdiry = 0
        elif sy > fy:
            fdiry = 1
        else:
            fdiry = 2

        if sx == 1:
            wx = 1
        elif sx == width:
            wx = 2
        else:
            wx = 0  

        if sy == 1:
            wy = 1 
        elif sy == height:
            wy = 2    
        else:
            wy = 0

        for i in sbod:
            if i[0] == sx and i[1] == (sy -1):
                bt = 1
            if i[0] == sx  and i[1] == (sy + 1):
                bb = 1
            if i[1] == sy and i[0] == (sx - 1):
                bl = 1
            if i[1] == sy and i[0] == (sx + 1):
                br = 1
        
        return (fdirx, fdiry, wx, wy, bt,bb,bl, br)
