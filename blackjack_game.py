import gym
env = gym.make('Blackjack-v0')


from tkinter import *
import numpy as np
from PIL import Image, ImageTk

observation = env.reset()

### blacjack environment shit

# observation = env.reset()
# action = env.action_space.sample()
# observation, reward, done, info = env.step(action)
###

### setup gui structure
root = Tk()
root.title("HitAgent_io")
root.geometry('{}x{}'.format(500 , 350))

dealer_title = Label(root, text ='Dealers Hand', font = "Arial")  
dealer_title.pack() 
  
frame = Frame(root) 
frame.pack() 
  
middleframe = Frame(root) 
middleframe.pack( side = BOTTOM )

bottomframe = Frame(middleframe) 
bottomframe.pack( side = BOTTOM )

## dealer images

dealer_images = []

card_size = (75, 125)
img = Image.open('cards/JPEG/blue_back.jpg')
img = img.resize(card_size, Image.ANTIALIAS)
dealer_card = ImageTk.PhotoImage(img)

dealer_imgLabel = Label(frame, image=dealer_card)
dealer_images.append(dealer_imgLabel)

# ai_hand

ai_images = []

ai_title = Label(middleframe, text ='Ai\'s hand', font = "Arial")  
ai_title.pack() 

img = Image.open('cards/JPEG/blue_back.jpg')
img = img.resize(card_size, Image.ANTIALIAS)
temp_image = ImageTk.PhotoImage(img)

ai_imgLabel = Label(middleframe, image=temp_image)
ai_images.append(ai_imgLabel)

obs_label = Label(bottomframe, text ='none', font = "Arial")  
obs_label.pack(side=BOTTOM) 
  
### Game functions

def getAiHand():
    print("hand:" , env.player)
    return env.player

### Button scaffold - needs to go here because this must be defined before beinng called
playRoundBtn = Button(bottomframe, text='Next', command=lambda:onclick(frame,middleframe,ai_images,dealer_images))

game_switcher = {
    "getAiHand" : getAiHand
}

# Gui functions

def addCardImage(path,frame, img_list):
    img = Image.open(path)
    img = img.resize(card_size, Image.ANTIALIAS)
    temp_image = ImageTk.PhotoImage(img)
    image_label = Label(frame,image=temp_image)
    img_list.append(image_label)
    return img_list

def updateLabelText(label,text):
    label["text"] = text

def getCardPath(card_index):
    suite = ["H", "D", "C", "S"]
    tens = ["K","Q","J","10"]
    path = ""
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

def updateAI(curr,middleframe, ai_images):
    path = getCardPath(curr)
    addCardImage(path, middleframe, ai_images)
    return ai_images

def updateDealer(curr, frame, ai_images):
    path = getCardPath(curr)
    addCardImage(path, frame, dealer_images)
    return dealer_images

gui_switcher = {
    "updateLabelText" : updateLabelText,
    "addCardImage" : addCardImage
}

gui_switcher = {
    "updateLabelText" : updateLabelText,
    "addCardImage" : addCardImage,
    "updateAI" : updateAI,
    "updateDealer" : updateDealer
}

def onclick(frame, middleframe, ai_images, dealer_images):
    curr_ai = env.player
    curr_dealer = env.dealer
    ai_images = updateAI(curr_ai[0], middleframe,ai_images)
    dealer_images = updateDealer(curr_dealer[0], frame, dealer_images)
    return (ai_images, dealer_images)
    
### displaying the button
for image in dealer_images:
    image.pack(side=LEFT)
for image in ai_images:
    image.pack(side=LEFT)

playRoundBtn.pack(side=LEFT)
root.mainloop()


# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         # env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print("obs", observation, "reward", reward, "done", done,"info",info)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()