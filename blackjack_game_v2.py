from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import gym
import time

import gym
import numpy as np
import pandas as pd
from collections import namedtuple
from collections import defaultdict

from tqdm import trange

images = []

#http://ernie55ernie.github.io/assets/blackjack.html

class Window(Frame):

    def __init__(self, master, pack):
        self.frame = Frame.__init__(self, master, height = 300, width = 150,relief="raised", background = "dark green")
        self.master = master
        self.player_cards = Text(self,height = 10, width = 50,background = "dark green")
        self.start_window()
        self.pack = pack

        
    def reset(self):
        self.player_cards.destroy()
        self.player_cards = Text(self,height = 10, width = 50,background = "dark green")
        images = []

    def getCardPath(self,card_index):
        suite = ["H", "D", "C", "S"]
        tens = ["K","Q","J","10"]
        path = ""
        if card_index == 0:
            path = "cards/JPEG/0.jpg"
            return path
        if card_index > 1:
            if card_index == 10:
                t_i = np.random.randint(3)
                s_i = np.random.randint(3)
                path = "cards/JPEG/" + tens[t_i] + suite[s_i] + ".jpg"
                return path
            else:
                s_i = np.random.randint(3)
                path = "cards/JPEG/" + str(card_index) + suite[s_i] + ".jpg"
                return path
        else:
            s_i = np.random.randint(3)
            path = "cards/JPEG/" + str(card_index) + "A"+ suite[s_i] + ".jpg"
            return path

        return "unexpected item in the bagging area"

    def update(self, card):
        im = []
        im.append(self.getCardPath(card))
        x = 0
        for i in (im):
            card_size = (75, 125)
            imgs = Image.open(im[x])
            imgs = imgs.resize(card_size, Image.ANTIALIAS)
            mi = ImageTk.PhotoImage(imgs)
            images.append(mi)
            self.player_cards.image_create(END, image = mi)
            x = x + 1
        self.player_cards.pack()
        self.player_cards.config(state=DISABLED)

    def start_window(self):
        self.pack()

root = Tk()
root.title("HitAgent_io")
env = gym.make('Blackjack-v0')

label1 = Label(root, text="Player", font="bold").pack()
sum1 = Label(root, text="Sum:")
sum1.pack()
player = Window(root,(1,0))
label2 = Label(root, text="Dealer", font="bold").pack()
sum2 = Label(root, text="Sum:")
sum2.pack()
dealer = Window(root,(1,1))

player.update(env.player[0])
dealer.update(env.dealer[0])

def updateLabelText(label,text):
    label["text"] = text

def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

def next(sum1, sum2):

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)

    discount_factor = 1.0

    for i in trange(100000):
        time.sleep(0.5)
        observation = env.reset()
        episodes = []
        for t in range(100):
            player.reset()
            dealer.reset()
            root.update_idletasks()
            action = sample_policy(observation)
            next_observation, reward, done, _ = env.step(action)
            for card in env.player:
                player.update(card)
            for card in env.dealer:
                dealer.update(card)
            episodes.append((observation, action, reward))

            updateLabelText(sum1,"Sum:" + str(observation[0]))
            updateLabelText(sum2,"Sum:" + str(np.sum(env.dealer)))

            root.update_idletasks()

            if done:
                break
            observation = next_observation


            # obtain unique observation set
    observations = set([x[0] for x in episodes])
    for i, observation in enumerate(observations):
        # first occurence of the observation
        idx = episodes.index([episode for episode in episodes if episode[0] == observation][0])
        
        Q = sum([episode[2] * discount_factor ** i for episode in episodes[idx:]])
        
        returns_sum[observation] += Q
        returns_count[observation] += 1.0
        
        V[observation] = returns_sum[observation] / returns_count[observation]
    return V

playRoundBtn = Button(root, text='Next', command=lambda:next(sum1, sum2)).pack()
# resetDealerBtn = Button(root, text='Reset Dealer', command=lambda:resetDealer()).pack()
# resetPlayerBtn = Button(root, text='Reset Player', command=lambda:resetPlayer()).pack()

root.mainloop()