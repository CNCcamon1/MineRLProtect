class NavTable:
    def __init__(self):
        self.table = [[]]
        self.current_episode = [[]]

    def insert_node(self, current_state, action, next_state, confidence):
        new_row = [current_state, action, next_state, confidence]
        self.current_episode.append(new_row)

    def get_action_by_state(self, state):
        current_best_confidence = 0
        for row in self.table:
            if (row[0] == state):
                if (row[3] >= current_best_confidence):
                    current_best = row[3]
        

        return current_best

    def adjust_confidence(self, reward):
        current_reward = reward
        for action in reversed(self.current_episode):
            action[3] += current_reward
            current_reward /= 2

    def add_episode_to_table(self):
        for action in self.current_episode:
            self.table.append(action)



    