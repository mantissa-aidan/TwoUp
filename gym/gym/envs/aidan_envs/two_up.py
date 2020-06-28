import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

def flip(np_random):
    return 1 if np_random.uniform() < 0.5 else 0

def subset_sum(numbers, target, partial=[], partial_sum=0):
    if partial_sum <= target:
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target, partial + [n], partial_sum + n)

def bets_list(notes, max):
    bets = []
    for bet in list(subset_sum(notes,max)):
        bets.append(sum(bet))
    return bets

class TwoUp(gym.Env):
    """Two up: two coin=s, one heads up, one tails up, 
    if it lands one heads up one tails up noone wins
    
    If you get two tails or two heads someone wins.

    Lets say 10 players plus the agent, all initialized with random amounts between
    $10-$250

    Can choose to bid heads or tails, or match a bid of another player
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_wealth=200, max_wealth=10000, max_rounds=10):
        super(TwoUp, self).__init__()
        #change this encoding so its just 1 discrete space
        self.action_space = spaces.Discrete(32)
        self.reward_range = (0, max_wealth)
        self.wealth = initial_wealth
        self.initial_wealth = initial_wealth
        self.winnings = 0
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(1, 1), dtype=np.float32)
        self.max_rounds = max_rounds
        self.max_wealth = max_wealth
        self.rounds = max_rounds
        self.side = ""
        self.coin1 = ""
        self.coin2 = ""
        self.round_result = ""
        self.heads_tails = ["H", "T"]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # if not action:
            # print("something fucked up")
        
        bet_index = action

        bets_heads = bets_list([10,20,50,100], 250)
        bets_tails = bets_list([10,20,50,100], 250)

        bets = bets_heads + bets_tails
        bet_in_dollars = bets[action]
        self.side = ""

        if action <= 15:
            self.side = "H"
        else:
            self.side = "T"

        self.rounds -= 1

        self.coin1 = self.heads_tails[flip(self.np_random)]
        self.coin2 = self.heads_tails[flip(self.np_random)]

        if(self.coin1 == self.coin2 and self.coin1 == self.side):
            self.round_result = "WIN"
            self.wealth += bet_in_dollars
            self.winnings = bet_in_dollars
        elif(self.coin1 == self.coin2 and self.coin1 != self.side):
            self.round_result = "LOSE"
            self.wealth -= bet_in_dollars
            self.winnings = bet_in_dollars * -1
        else:
            self.round_result = "NO WINNERS"
        done = self.wealth < 0.01 or self.wealth == self.max_wealth or not self.rounds
        reward = self.winnings if done else 0.0

        return self._get_obs(), reward, done, {}

    def get_bets(self):
        return self.betting_pool

    def reset(self):
        self.rounds = self.max_rounds
        self.wealth = self.initial_wealth
        self.winnings = 0
        return self._get_obs()

    def render(self, mode='human'):
        print("Flip:", self.coin1, self.coin2, "Side bet", self.side, "Result:", self.round_result, "Winnings: ", self.winnings, "Current wealth: ", self.wealth, "; Rounds left: ", self.rounds)
        print()

    def _get_obs(self):
        return (self.winnings)

    def reset(self):
        # re-init everything to draw new parameters etc, but preserve the RNG for
        # reproducibility and pass in the same hyper-parameters as originally specified:
        self.__init__()
        return self._get_obs()


