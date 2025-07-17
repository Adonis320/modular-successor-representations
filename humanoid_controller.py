class Controller():
    def __init__(self):
        self.direction_index_1 = 0
        self.previous_step_1 = 0

        self.direction_index_2 = 0
        self.previous_step_2 = 0

        self.direction_index_3 = 0
        self.previous_step_3 = 0

        self.direction_index_4 = 0
        self.previous_step_4 = 0
        UP = 3
        DOWN = 1
        LEFT = 0
        RIGHT = 2

        self.DIRECTIONS_1_1 = [LEFT, DOWN, RIGHT, UP]
        self.DIRECTIONS_1_2 = [DOWN, LEFT, UP, RIGHT]

        self.DIRECTIONS_2_1 = [RIGHT, UP, LEFT, DOWN]
        self.DIRECTIONS_2_2 = [UP, RIGHT, DOWN, LEFT]


        self.DIRECTIONS_3_1 = [DOWN, RIGHT, UP, LEFT]
        self.DIRECTIONS_3_2 = [RIGHT, DOWN, LEFT, UP]

        self.DIRECTIONS_4_1 = [UP, LEFT, DOWN, RIGHT]
        self.DIRECTIONS_4_2 = [LEFT, UP, RIGHT, DOWN]

    def reset(self):
        self.direction_index_1 = 0
        self.previous_step_1 = 0

        self.direction_index_2 = 0
        self.previous_step_2 = 0

        self.direction_index_3 = 0
        self.previous_step_3 = 0

        self.direction_index_4 = 0
        self.previous_step_4 = 0

    def get_action(self, reverse):
        if reverse:
            action_1 = self.DIRECTIONS_1_2[self.direction_index_1]
            action_2 = self.DIRECTIONS_2_2[self.direction_index_2]
            action_3 = self.DIRECTIONS_3_2[self.direction_index_3]
            action_4 = self.DIRECTIONS_4_2[self.direction_index_4]

        else:
            action_1 = self.DIRECTIONS_1_1[self.direction_index_1]
            action_2 = self.DIRECTIONS_2_1[self.direction_index_2]
            action_3 = self.DIRECTIONS_3_1[self.direction_index_3]
            action_4 = self.DIRECTIONS_4_1[self.direction_index_4]
        return action_1, action_2, action_3, action_4
    
    def update(self, reward):
        if reward[1] != -110:
            self.previous_step_1 +=1
        self.direction_index_1 = (self.previous_step_1 // 4) % 4

        if reward[2] != -110:
            self.previous_step_2 +=1
        self.direction_index_2 = (self.previous_step_2 // 4) % 4

        if reward[3] != -110:
            self.previous_step_3 +=1
        self.direction_index_3 = (self.previous_step_3 // 4) % 4

        if reward[4] != -110:
            self.previous_step_4 +=1
        self.direction_index_4 = (self.previous_step_4 // 4) % 4