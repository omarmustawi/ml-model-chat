from collections import defaultdict
import random


class FeedbackBandit:

    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon
        self.counts = defaultdict(int)
        self.values = defaultdict(float)

    def choose(self, candidates):
        if len(candidates) == 0:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(candidates)
        

        return max(candidates, key=lambda x: self.values[x])
    

    def update(self, meal_name, reward):
        self.counts[meal_name] += 1
        n = self.counts[meal_name]
        old = self.values[meal_name]
        self.values[meal_name] = old + (reward - old) / n

