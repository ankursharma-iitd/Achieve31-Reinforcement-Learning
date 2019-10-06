#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# for vectors manipulation
import numpy as np

# for plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# visualize plots in the jupyter notebook
# check more https://goo.gl/U3Ai8R
get_ipython().run_line_magic('matplotlib', 'inline')

# for generating random values
import random

# for representing things like card value or colors
from enum import Enum  

# for copying python objects
import copy

from tqdm import tqdm_notebook, tqdm

from pylab import rcParams


# ### Plotting Method

# In[ ]:


#method adapted from Easy21 assignment of https://github.com/analog-rl/Easy21
def plot_value_function(agent_type, value_function, method, title='Value Function', usables=[0,0,0], generate_gif=False, train_steps=None, save=None, transpose=True):
    """
    Plots a value function as a surface plot, like in: https://goo.gl/aF2doj

    You can choose between just plotting the graph for the value function
    which is the default behaviour (generate_gif=False) or to train the agent
    a couple of times and save the frames in a gif as you train.

    Args:
        agent: An agent.
        title (string): Plot title.
        generate_gif (boolean): If want to save plots as a gif.
        train_steps: If is not None and generate_gif = True, then will use this
                     value as the number of steps to train the model at each frame.
    """
    # you can change this values to change the size of the graph
    title += ' (' + str(train_steps) + ' Episodes, Usables ' + str(usables) + ', ' + method + ')'
    title = agent_type + ' ' + title
    fig = plt.figure(title, figsize=(10, 5))
    
    # explanation about this line: https://goo.gl/LH5E7i
    ax = fig.add_subplot(111, projection='3d')
    
    if transpose:
        V = np.transpose(value_function[:,:,usables[0],usables[1],usables[2]])
    else:
        V = value_function[:,:,usables[0],usables[1],usables[2]]
    
    if generate_gif:
        print('gif will be saved as %s' % title)
    
    def plot_frame(ax):
        # min value allowed accordingly with the documentation is 1
        # we're getting the max value from V dimensions
        min_x = 1
        max_x = V.shape[0]
        min_y = 1
        max_y = V.shape[1]

        # creates a sequence from min to max
        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)

        # creates a grid representation of x_range and y_range
        X, Y = np.meshgrid(x_range, y_range)

        # get value function for X and Y values
        def get_stat_val(x, y):
            return V[x, y]
        Z = get_stat_val(X, Y)

        # creates a surface to be ploted
        # check documentation for details: https://goo.gl/etEhPP
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('Value')
        return ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, 
                               linewidth=0, antialiased=False)
    
    def animate(frame):
        # clear the plot and create a new surface
        ax.clear()
        surf = plot_frame(ax)
        # if we're going to generate a gif we need to train a couple of times
        if generate_gif:
            i = agent.iterations
            # cool math to increase number of steps as we go
            if train_steps is None:
                step_size = int(min(max(1, agent.iterations), 2 ** 16))
            else:
                step_size = train_steps

            agent.train(step_size)
            plt.title('%s MC score: %s frame: %s' % (title, float(agent.wins)/agent.iterations*100, frame))
        else:
            plt.title(title)

        fig.canvas.draw()
        return surf

    ani = animation.FuncAnimation(fig, animate, 32, repeat=False)

    # requires gif writer
    if generate_gif:
        ani.save(title + '.gif', writer='imagemagick', fps=3)
    else:
        if save is None:
            plt.show()
            plt.close()
        else:
            plt.savefig(save)
            plt.close('all')


# ### Basic Class for Color, Action, Card, Deck, Player, Dealer, State, etc.

# In[ ]:


class Color(Enum):
    BLACK = 0
    RED = 1
    
class Action(Enum):
    HIT = 0
    STICK = 1


# In[ ]:


class Card(object):
    def __init__(self, color=None, value=None):
        if value is not None:
            self.value = value
        else:
            self.value = random.randint(1, 10)
        if color == Color.BLACK or color == Color.RED:
            self.color = color
        else:
            random_num = random.random()
            if(random_num <= (1.0/3)):
                self.color = Color.RED
            else:
                self.color = Color.BLACK
                
    def _print_card_(self):
        print("Color: {}, Value: {}".format(self.color, self.value))


# In[ ]:


class Deck(object):
    def sample_card(self, color=None, value=None):
        return Card(color, value)


# In[ ]:


# State characterised by (agent_sum, dealer_sum, Usable1, Usable2, Usable3)
# Usable1 = 0 -> there is no usable 1 in the current hand
class State(object):
    def __init__(self, agent_sum=0, show_card=0, agent_usables=[0,0,0], dealer_sum=0, dealer_usables=[0,0,0], is_terminal=False):
        self.show_card = show_card
        self.agent_sum = agent_sum
        self.dealer_sum = dealer_sum
        self.agent_usables = agent_usables
        self.dealer_usables = dealer_usables
        self.is_terminal = is_terminal
        
    def get_tuple(self):
        usables = self.agent_usables
        return (self.agent_sum, self.show_card - 4, usables[0], usables[1], usables[2])
        
    def _print_state_(self):
        print("Dealer-Face-Card: ({}), Agent-Sum: ({}), Dealer-Sum: ({}), Usables: ({}, {}), Terminal: ({})".format(
                                                                                            self.show_card,\
                                                                                            self.agent_sum,\
                                                                                            self.dealer_sum,\
                                                                                            self.agent_usables,\
                                                                                            self.dealer_usables,\
                                                                                            self.is_terminal))


# In[ ]:


class Player(object):
    def policy(self, state):
        raise NotImplemented()

class Dealer(object):
    def policy(self, state):
        if(state.dealer_sum >= 25):
            return Action.STICK
        else:
            return Action.HIT


# ### Define Environment Class

# In[ ]:


class Environment(object):
    def __init__(self):
        self.deck = Deck()
        self.dealer = Dealer()
        self.agent_max_value = 32  # max value an agent can get during the game
        self.dealer_max_value = 10  # max value the dealer can get when taking the first card
        self.usable_1_values = 3 # can take states 0(not-present), 1(usable) or 2(non-usable) for card-value 1
        self.usable_2_values = 3 # can take states 0(not-present), 1(usable) or 2(non-usable) for card-value 2
        self.usable_3_values = 3 # can take states 0(not-present), 1(usable) or 2(non-usable) for card-value 3
        self.actions_count = 2  # number of possible actions in each state
        
    def check_bust(self, current_sum):
        return (current_sum < 0) or (current_sum > 31)
    
    # get reward when both of them have sticked without going bust
    def get_reward_bust(self, state):
        if(state.agent_sum > state.dealer_sum):
            return 1
        elif(state.agent_sum == state.dealer_sum):
            return 0
        return -1
    
    def sample_card_value(self, color=None, value=None):
        card = self.deck.sample_card(color, value)
        if(card.color == Color.BLACK):
            return card.value
        else:
            return -card.value
    
    def update_dealer_usables(self, state, card_value):
        if(card_value == 1 or card_value == 2 or card_value == 3):
            idx = card_value - 1
            old_state = state.dealer_usables[idx]
            if(old_state == 0):
                # previously not present
                # value can either be new_value or (new_value + 10)
                if self.check_bust(state.dealer_sum + card_value + 10):
                    # going bust when using (new_value + 10)
                    # make it non-usable
                    new_state = 2
                else:
                    # can use it as a usable card, add 10 and make it usable
                    card_value += 10
                    new_state = 1
            elif(old_state == 1):
                # previously used as a usable
                new_state = 2
            else:
                new_state = 2
            state.dealer_usables[idx] = new_state
        return state, card_value
    
    def update_agent_usables(self, state, card_value):
        if(card_value == 1 or card_value == 2 or card_value == 3):
            idx = card_value - 1
            old_state = state.agent_usables[idx]
            if(old_state == 0):
                # previously not present
                # value can either be new_value or (new_value + 10)
                if self.check_bust(state.agent_sum + card_value + 10):
                    # going bust when using (new_value + 10)
                    # make it non-usable
                    new_state = 2
                else:
                    # can use it as a usable card, add 10 and make it usable
                    card_value += 10
                    new_state = 1
            elif(old_state == 1):
                # previously used as a usable
                new_state = 2
            else:
                new_state = 2
            state.agent_usables[idx] = new_state
        return state, card_value
        
    # play dealer's turn when the agent has sticked    
    def play_dealer(self, state):
        while(True):
            action = self.dealer.policy(state)
            if(action == Action.HIT):
                new_card_value = self.sample_card_value()
                state, new_card_value = self.update_dealer_usables(copy.deepcopy(state), new_card_value)
                state.dealer_sum += new_card_value
            state.is_terminal = self.check_bust(state.dealer_sum)
            if(state.is_terminal or (action == Action.STICK)):
                break
        return state
        
    # Both the dealer and agent take one card each, dealer's card is visible to the agent
    def initial_state(self):
        agent_card_value = self.sample_card_value()
        dealer_card_value = self.sample_card_value()
        state = State(agent_usables=[0,0,0], dealer_usables=[0,0,0])
        state, agent_card_value = self.update_agent_usables(copy.deepcopy(state), agent_card_value)
        state, dealer_card_value = self.update_dealer_usables(copy.deepcopy(state), dealer_card_value)
        state.agent_sum += agent_card_value
        state.dealer_sum += dealer_card_value
        state.show_card = dealer_card_value
        agent_busted = self.check_bust(state.agent_sum)
        dealer_busted = self.check_bust(state.show_card)
        if (agent_busted or dealer_busted):
            state.is_terminal = True
        else:
            state.is_terminal = False
        return state
    
    # Given (state, action) return (next_state, reward)
    def step(self, state, action):
        agent_sum = state.agent_sum
        show_card = state.show_card
        agent_usables = state.agent_usables
        reward = 0
        next_state = copy.deepcopy(state)
        if(state.is_terminal):
            print('Cannot take action on a terminal state')
        if action == Action.STICK:
            next_state = self.play_dealer(copy.deepcopy(state))
            if next_state.is_terminal:
                reward = 1
            else:
                next_state.is_terminal = True
                reward = self.get_reward_bust(next_state)
        else:
            agent_card_value = self.sample_card_value()
            next_state, agent_card_value = self.update_agent_usables(copy.deepcopy(state), agent_card_value)
            next_state.agent_sum += agent_card_value
            next_state.is_terminal = self.check_bust(next_state.agent_sum)
            if next_state.is_terminal:
                reward = -1
        return next_state, reward


# ### General Agent (Base Class for all Agents Learnt)

# In[ ]:


class Agent(Player):
    def __init__(self, environment, discount_factor=1.0):
        Player.__init__(self)
        self.env = environment
        self.discount_factor = discount_factor
        self.value_function = np.zeros([self.env.agent_max_value,                                         self.env.dealer_max_value,                                         self.env.usable_1_values,                                         self.env.usable_2_values,                                         self.env.usable_3_values])
        self.action_value_function = np.zeros([self.env.agent_max_value,                                               self.env.dealer_max_value,                                               self.env.usable_1_values,                                               self.env.usable_2_values,                                               self.env.usable_3_values,                                               self.env.actions_count])
        self.matches_won = 0.0
        self.matches_draw = 0.0
        self.matches_lose = 0.0
        self.num_games_played = 0.0
    
    def get_value_function(self):
        for i in range(self.env.agent_max_value):
            for j in range(self.env.dealer_max_value):
                for k in range(self.env.usable_1_values):
                    for l in range(self.env.usable_2_values):
                        for m in range(self.env.usable_3_values):
                            s = State(i,j+4,[k,l,m])
                            action = self.take_greedy_action(s)
                            satup = s.get_tuple() + (action.value, )
                            self.value_function[s.get_tuple()] = self.action_value_function[satup]
        return self.value_function
    
    def get_action_value_function(self):
        return self.action_value_function
        
    def take_random_action(self):
        num = random.random()
        if num <= 0.5:
            return Action.HIT
        return Action.STICK
    
    def take_greedy_action(self, state):
        stup = state.get_tuple()
        Q = self.get_action_value_function()
        return Action(np.argmax(Q[stup]))


# ### Monte Carlo Evaluation

# In[ ]:


class MCAgentEvaluation(Agent):
    def __init__(self, environment, discount_factor=1):
        Agent.__init__(self, environment, discount_factor)
        self.visit_count_V = np.zeros([self.env.agent_max_value,                             self.env.dealer_max_value,                             self.env.usable_1_values,                             self.env.usable_2_values, 
                            self.env.usable_3_values])
        self.visit_count_Q = np.zeros([self.env.agent_max_value,                            self.env.dealer_max_value,                            self.env.usable_1_values,                            self.env.usable_2_values,                            self.env.usable_3_values,                            self.env.actions_count])
        
    
    def predict_V(self, episode, method='all-visit'):
        T = len(episode)
        gamma = self.discount_factor
        last_reward = episode[T - 1][2]
        visit_set = set()
        for ep_num, (s, a, r) in enumerate(episode):
            stup = s.get_tuple()
            G = (gamma ** (T - 1 - ep_num)) * last_reward * 1.0
            k = self.visit_count_V[stup]
            mean_old = self.value_function[stup]
            if method == 'first-visit':
                if not stup in visit_set:
                    self.value_function[stup] = (1.0 * ((k * mean_old) + G))/(k + 1) 
                    self.visit_count_V[stup] = k + 1
                    visit_set.add(stup)
            else:
                self.value_function[stup] = (1.0 * ((k * mean_old) + G))/(k + 1) 
                self.visit_count_V[stup] = k + 1
                
    def predict_Q(self, episode, method='all-visit'):
        T = len(episode)
        gamma = self.discount_factor
        last_reward = episode[T - 1][2]
        visit_set = set()
        for ep_num, (s, a, r) in enumerate(episode):
            stup = s.get_tuple()
            satup = stup + (a.value,)
            G = (gamma ** (T - 1 - ep_num)) * last_reward * 1.0
            k = self.visit_count_Q[satup]
            mean_old = self.action_value_function[satup]
            if method == 'first-visit':
                if not satup in visit_set:
                    self.action_value_function[satup] = (1.0 * ((k * mean_old) + G))/(k + 1) 
                    self.visit_count_Q[satup] = k + 1
                    visit_set.add(satup)
            else:
                self.action_value_function[satup] = (1.0 * ((k * mean_old) + G))/(k + 1) 
                self.visit_count_Q[satup] = k + 1
        
    
    def policy(self, state):
        if(state.agent_sum >= 25):
            return Action.STICK
        else:
            return Action.HIT
        
    def generate_episode(self, es=False):
        episode = []
        state = self.env.initial_state()
#         if(state.is_terminal):
#             print('terminal state')
#             state._print_state_()
        while not state.is_terminal:
            
            # choose action as per agent's policy
            if es:
                action = self.take_random_action()
                es = False
            else:
                action = self.policy(state)

            # execute action in the env and gather rewards
            next_state, reward = self.env.step(copy.deepcopy(state), action)

            # store the episode
            episode.append((copy.deepcopy(state), action, reward))

            # update the state
            state = next_state
            
        return episode
    
    def train_V(self, num_episodes, method='all-visit'):
        epi_num = 0
        pbar = tqdm(total=num_episodes)
        while epi_num < num_episodes:
            episode = self.generate_episode(es=False)
            if(len(episode) == 0):
                continue
            self.num_games_played += 1.0
            self.predict_V(episode, method)
#             if epi_num % 100 == 0 and epi_num != 0:
#                 print("Episode: %d" % epi_num)
            pbar.update(1)
            epi_num += 1
        pbar.close()
        return self.value_function
    
    def train_Q(self, num_episodes, method='all-visit'):
        epi_num = 0
        pbar = tqdm(total=num_episodes)
        while epi_num < num_episodes:
            episode = self.generate_episode(es=True)
            if(len(episode) == 0):
                continue
            self.num_games_played += 1.0
            self.predict_Q(episode, method)
#             if epi_num % 100 == 0 and epi_num != 0:
#                 print("Episode: %d" % epi_num)
            pbar.update(1)
            epi_num += 1
        pbar.close()
        return self.action_value_function


# In[ ]:


def train_and_save_MC(num_episodes, method):
    environment = Environment()
    mc_agent = MCAgentEvaluation(environment)
    V = mc_agent.train_V(num_episodes, method)
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                save_path = './MC/' + str(num_episodes) + '-' + str(i) + '-' + str(j) + '-' + str(k) + '-' + method + '.png'
                plot_value_function('MC', V, method=method, train_steps=num_episodes, usables=[i,j,k], save=save_path)


# In[ ]:


counts = [100, 100000, 1000000, 10000000]
methods = ['all-visit', 'first-visit']
for count in counts:
    for method in methods:
        train_and_save_MC(count, method)


# In[ ]:


environment = Environment()
mc_agent = MCAgentEvaluation(environment)
V = mc_agent.train_V(100000, 'all-visit')
plot_value_function('MC', V, method='all-visit', train_steps=100000, usables=[0,0,0])


# In[ ]:


environment = Environment()
mc_agent = MCAgentEvaluation(environment)
num_episodes = 100000
method = 'all-visit'
V = mc_agent.train_V(num_episodes, method)


# In[ ]:


plot_value_function(mc_agent.get_value_function(), method=method, train_steps=num_episodes, usables=[0,0,0])


# In[ ]:


out_Q = mc_agent.train_Q(1000, 'first-visit')


# ## k-Step TD Evaluation

# In[ ]:


class kTDEvaluation(Agent):
    def __init__(self, environment, discount_factor=1, alpha=0.1, k=1):
        Agent.__init__(self, environment, discount_factor)
        self.alpha =  alpha
        self.k = k
    
    def policy(self, state):
        if(state.agent_sum >= 25):
            return Action.STICK
        else:
            return Action.HIT
        
    def train_V(self, num_episodes=100):
        pbar = tqdm(total=num_episodes)
        for i in range(num_episodes):
            
            # get a non-terminal state S0
            state = self.env.initial_state()
            while(state.is_terminal):
                state = self.env.initial_state()

            # store_states
            states_seen = {}
            states_seen[0] = state.get_tuple()

            # initialise T <- infinity
            T = float('inf')
            t = 0
            last_reward = 0
            k = self.k
            tau = 1 - k
            gamma = self.discount_factor
            alpha = self.alpha
    
            while(tau != (T - 1)):
                if t < T:
                    if not state.is_terminal:
                        action = self.policy(state)
                        next_state, reward = self.env.step(copy.deepcopy(state), action)
                        states_seen[t + 1] = next_state.get_tuple()
                        state = next_state
                    if(next_state.is_terminal):
                        T = t + 1
                        last_reward = reward
                tau = t - k + 1
                if tau >= 0:
                    G = 0
                    if T != float('inf'):
                        G = ((gamma ** (T - tau - 1)) * last_reward)
                    if (tau + k < T):
                        stup = states_seen[tau + k]
                        G += ((gamma ** k) * self.value_function[stup])
                    stup_to_update = states_seen[tau]
                    self.value_function[stup_to_update] += (alpha * (G - self.value_function[stup_to_update]))
                t += 1
            pbar.update(1)
        pbar.close()
        return self.value_function
    
    def train_Q(self, num_episodes=100):
        pbar = tqdm(total=num_episodes)
        for i in range(num_episodes):
            
            # get a non-terminal state S0
            state = self.env.initial_state()
            while(state.is_terminal):
                state = self.env.initial_state()

            # store_states
            states_seen = {}
            action = self.take_random_action()
            states_seen[0] = state.get_tuple() + (action.value,)

            # initialise T <- infinity
            T = float('inf')
            t = 0
            last_reward = 0
            k = self.k
            tau = 1 - k
            gamma = self.discount_factor
            alpha = self.alpha
    
            while(tau != (T - 1)):
                if t < T:
                    next_state, reward = self.env.step(copy.deepcopy(state), action)
                    if not next_state.is_terminal:
                        next_action = self.policy(next_state)
                        satup = next_state.get_tuple() + (next_action.value,)
                        states_seen[t + 1] = satup
                        state = next_state
                        action = next_action
                    if(next_state.is_terminal):
                        T = t + 1
                        last_reward = reward
                tau = t - k + 1
                if tau >= 0:
                    G = 0
                    if T != float('inf'):
                        G = ((gamma ** (T - tau - 1)) * last_reward)
                    if (tau + k < T):
                        stup = states_seen[tau + k]
                        G += ((gamma ** k) * self.action_value_function[stup])
                    stup_to_update = states_seen[tau]
                    self.action_value_function[stup_to_update] += (alpha * (G - self.action_value_function[stup_to_update]))
                t += 1
            pbar.update(1)
        pbar.close()
        return self.action_value_function


# In[ ]:


def train_and_save_TD(num_episodes, k, save=False):
    environment = Environment()
    td_agent = kTDEvaluation(environment, alpha=0.1, k=k)
    V = td_agent.train_V(num_episodes)
    if save:
        for i in range(0,3):
            for j in range(0,3):
                for kk in range(0,3):
                    save_path = './TD/' + str(k) + '-' + str(num_episodes) + '-' + str(i) + '-' + str(j) + '-' + str(kk) + '.png'
                    plot_value_function('TD', V, method='k='+str(k), train_steps=num_episodes, usables=[i,j,kk], save=save_path)
    return V


# In[ ]:


all_episodes = [25, 250]
ks = [1, 3, 5, 10, 100, 1000]
num_episodes = 5000
for k in ks:
    for episode_count in all_episodes:
        V = np.zeros([32,10,3,3,3])
        for i in range(episode_count):
            V += train_and_save_TD(num_episodes, k, False)
            final_V = V / episode_count
        for i in range(0,3):
            for j in range(0,3):
                for kk in range(0,3):
                    save_path = './TD/' + 'average-' + str(k) + '-' + str(episode_count) + '-' + str(num_episodes) + '-' + str(i) + '-' + str(j) + '-' + str(kk) + '.png'
                    plot_value_function('TD', final_V, method='k='+str(k), train_steps=10000, usables=[i,j,kk], save=save_path)


# In[ ]:


environment = Environment()
num_episodes = 10000
k = 3
td_agent = kTDEvaluation(environment, alpha=0.1, k=k)
V = td_agent.train_V(num_episodes)


# In[ ]:


stup = (0,0,0)


# In[ ]:


V[stup].shape


# In[ ]:


environment = Environment()
num_episodes = 10000
k = 3
td_agent = kTDEvaluation(environment, alpha=0.1, k=k)
Q = td_agent.train_Q(num_episodes)


# In[ ]:


V[:,:,0,0,0]


# ### k-step SARSA Agent (Fixed or Decaying Epsilon)

# In[ ]:


class kSarsaControl(Agent):
    def __init__(self, environment, alpha=0.1, k=1, discount_factor=1):
        Agent.__init__(self, environment, discount_factor)
        self.alpha =  alpha
        self.k = k
    
    def ep_greedy_policy(self, state, epsilon):
        num = random.random()
        stup = state.get_tuple()
        Q = self.get_action_value_function()
        if(num <= epsilon):
            action = self.take_random_action()
        else:
            action = Action(np.argmax(Q[stup]))
        return action
    
    def policy(self, state):
        return self.take_greedy_action(state)
    
    def generate_test_episode(self, epsilon=0.1, es=True):
        episode = []
        
        state = self.env.initial_state()
        while(state.is_terminal):
            state = self.env.initial_state()
        
        while not state.is_terminal:
            
            # choose action as per agent's policy
            action = self.policy(state)

            # execute action in the env and gather rewards
            next_state, reward = self.env.step(copy.deepcopy(state), action)

            # store the episode
            episode.append((copy.deepcopy(state), action, reward))

            # update the state
            state = next_state
            
        return episode
    
    
    def train_Q(self, num_episodes=100, epsilon=0.1, decay=False, get_total_reward=False):
        pbar = tqdm(total=num_episodes)
        total_reward = []
        for i in range(num_episodes):
            
            # get a non-terminal state S0
            state = self.env.initial_state()
            while(state.is_terminal):
                state = self.env.initial_state()

            # store_states
            states_seen = {}
            action = self.take_random_action()
            states_seen[0] = state.get_tuple() + (action.value,)

            # initialise T <- infinity
            T = float('inf')
            t = 0
            last_reward = 0
            k = self.k
            tau = 1 - k
            gamma = self.discount_factor
            alpha = self.alpha
            count = 1
    
            while(tau != (T - 1)):
                if t < T:
                    next_state, reward = self.env.step(copy.deepcopy(state), action)
                    if not next_state.is_terminal:
                        next_action = self.ep_greedy_policy(next_state, ((1.0 * epsilon)/count))
                        satup = next_state.get_tuple() + (next_action.value,)
                        states_seen[t + 1] = satup
                        state = next_state
                        action = next_action
                    if(next_state.is_terminal):
                        T = t + 1
                        last_reward = reward
                tau = t - k + 1
                if tau >= 0:
                    G = 0
                    if T != float('inf'):
                        G = ((gamma ** (T - tau - 1)) * last_reward)
                    if (tau + k < T):
                        stup = states_seen[tau + k]
                        G += ((gamma ** k) * self.action_value_function[stup])
                    stup_to_update = states_seen[tau]
                    self.action_value_function[stup_to_update] += (alpha * (G - self.action_value_function[stup_to_update]))
                    if decay:
                        count += 1
                t += 1
            total_reward += [last_reward]
            pbar.update(1)
        pbar.close()
        if get_total_reward:
            return self.get_action_value_function(), total_reward
        return self.get_action_value_function()


# In[ ]:


environment = Environment()
num_episodes = 100000
k = 10
sarsa_agent = kSarsaControl(environment, alpha=0.1, k=k)
Q = sarsa_agent.train_Q(num_episodes, decay=True)


# In[ ]:


plot_value_function('SARSA', sarsa_agent.get_value_function(), method='k='+str(k), train_steps=num_episodes, usables=[0,1,1])


# ### Q-Learning Agent

# In[ ]:


class QLearningAgent(Agent):
    def __init__(self, environment, alpha=0.1, discount_factor=1):
        Agent.__init__(self, environment, discount_factor)
        self.alpha =  alpha
        self.k = 1
    
    def ep_greedy_policy(self, state, epsilon):
        num = random.random()
        stup = state.get_tuple()
        Q = self.get_action_value_function()
        if(num <= epsilon):
            action = self.take_random_action()
        else:
            action = Action(np.argmax(Q[stup]))
        return action
    
    def policy(self, state):
        return self.take_greedy_action(state)
    
    def generate_test_episode(self, epsilon=0.1, es=True):
        episode = []
        
        state = self.env.initial_state()
        while(state.is_terminal):
            state = self.env.initial_state()
        
        while not state.is_terminal:
            
            # choose action as per agent's policy
            action = self.policy(state)

            # execute action in the env and gather rewards
            next_state, reward = self.env.step(copy.deepcopy(state), action)

            # store the episode
            episode.append((copy.deepcopy(state), action, reward))

            # update the state
            state = next_state
            
        return episode
    
    def generate_episode(self, epsilon=0.1, decay=False, es=True):
        episode = []
        
        state = self.env.initial_state()
        while(state.is_terminal):
            state = self.env.initial_state()
        
        count = 0
        while not state.is_terminal:
            
            # choose action as per agent's policy
            if es:
                action = self.take_random_action()
                es = False
            else:
                action = self.ep_greedy_policy(state, (1.0 * epsilon)/count)

            # execute action in the env and gather rewards
            next_state, reward = self.env.step(copy.deepcopy(state), action)

            # store the episode
            episode.append((copy.deepcopy(state), action, reward))

            # update the state
            state = next_state
            
            if decay:
                count += 1
            
        return episode
    
    def train_Q(self, num_episodes=100, epsilon=0.1, decay=False, get_total_reward=False):
        pbar = tqdm(total=num_episodes)
        total_reward = []
        for i in range(num_episodes):
            
            # get a non-terminal state S0
            state = self.env.initial_state()
            while(state.is_terminal):
                state = self.env.initial_state()

            # store_states
            states_seen = {}
            action = self.take_random_action()
            states_seen[0] = state.get_tuple() + (action.value,)

            # initialise T <- infinity
            T = float('inf')
            t = 0
            last_reward = 0
            k = self.k
            tau = 1 - k
            gamma = self.discount_factor
            alpha = self.alpha
            count = 1
    
            while(tau != (T - 1)):
                if t < T:
                    next_state, reward = self.env.step(copy.deepcopy(state), action)
                    if not next_state.is_terminal:
                        next_action = self.ep_greedy_policy(next_state, ((1.0 * epsilon)/count))
                        satup = next_state.get_tuple() + (next_action.value,)
                        states_seen[t + 1] = satup
                        state = next_state
                        action = next_action
                    if(next_state.is_terminal):
                        T = t + 1
                        last_reward = reward
                tau = t - k + 1
                if tau >= 0:
                    G = 0
                    if T != float('inf'):
                        G = ((gamma ** (T - tau - 1)) * last_reward)
                    if (tau + k < T):
                        stup = states_seen[tau + k]
                        state_tuple = stup[:-1]
                        # maximum of all the qvalues in that state
                        max_q_value = np.max(self.action_value_function[state_tuple])
                        G += ((gamma ** k) * max_q_value)
                    stup_to_update = states_seen[tau]
                    self.action_value_function[stup_to_update] += (alpha * (G - self.action_value_function[stup_to_update]))
                    if decay:
                        count += 1
                t += 1
            pbar.update(1)
            total_reward += [last_reward]
        pbar.close()
        if get_total_reward:
            return self.get_action_value_function(), total_reward
        return self.get_action_value_function()


# In[ ]:


environment = Environment()
num_episodes = 500000
qlearning_agent = QLearningAgent(environment, alpha=0.1)
Q = qlearning_agent.train_Q(num_episodes, decay=True)


# In[ ]:


plot_value_function('Q Learning', qlearning_agent.get_value_function(), method='k='+str(1), train_steps=num_episodes, usables=[0,0,0])


# ### Forward View Eligibility Traces

# In[ ]:


class Sarsa_lambda(Agent):
    def __init__(self, environment, alpha=0.1, discount_factor=1, _lambda=0.5):
        Agent.__init__(self, environment, discount_factor)
        self.alpha =  alpha
        self._lambda = _lambda
        # randomly initialised
        self.action_value_function = np.random.uniform(low=-1.0, high=1.0, size=
                                              [self.env.agent_max_value, \
                                              self.env.dealer_max_value, \
                                              self.env.usable_1_values, \
                                              self.env.usable_2_values, \
                                              self.env.usable_3_values, \
                                              self.env.actions_count])
        self.E = np.zeros([self.env.agent_max_value,                           self.env.dealer_max_value,                           self.env.usable_1_values,                           self.env.usable_2_values,                           self.env.usable_3_values,                           self.env.actions_count])
        
    def ep_greedy_policy(self, state, epsilon):
        num = random.random()
        stup = state.get_tuple()
        Q = self.get_action_value_function()
        if(num <= epsilon):
            action = self.take_random_action()
        else:
            action = Action(np.argmax(Q[stup]))
        return action
    
    def random_policy(self, state):
        return self.take_random_action()
    
    def policy(self, state):
        return self.take_greedy_action(state)
    
    def get_state_action_tuple(self, state, action):
        stup = state.get_tuple()
        return (stup + (action.value, ))
    
    def generate_test_episode(self, epsilon=0.1, es=True):
        episode = []
        
        state = self.env.initial_state()
        while(state.is_terminal):
            state = self.env.initial_state()
        
        while not state.is_terminal:
            
            # choose action as per agent's policy
            action = self.policy(state)

            # execute action in the env and gather rewards
            next_state, reward = self.env.step(copy.deepcopy(state), action)

            # store the episode
            episode.append((copy.deepcopy(state), action, reward))

            # update the state
            state = next_state
            
        return episode
    
    def generate_episode(self, epsilon=0.1, decay=True, es=True):
        episode = []
        
        state = self.env.initial_state()
        while(state.is_terminal):
            state = self.env.initial_state()
        
        count = 0
        while not state.is_terminal:
            
            # choose action as per agent's policy
            if es:
                action = self.take_random_action()
                es = False
            else:
                action = self.ep_greedy_policy(state, (1.0 * epsilon)/count)

            # execute action in the env and gather rewards
            next_state, reward = self.env.step(copy.deepcopy(state), action)

            # store the episode
            episode.append((copy.deepcopy(state), action, reward))

            # update the state
            state = next_state
            
            if decay:
                count += 1
            
        return episode
        
    
    def train_Q_forward(self, num_episodes, epsilon=0.1, decay=True):
        pbar = tqdm(total=num_episodes)
        for i in range(num_episodes):
            gamma = self.discount_factor
            alpha = self.alpha
            _lambda = self._lambda
            episode = self.generate_episode(epsilon, decay, es=True)
            T = len(episode)
            last_reward = episode[T - 1][2]
            for t, (s_t, a_t, _) in enumerate(episode):
                G_t_lambda = 0
                for k in range(1, T - t):
                    state, action, _ = episode[t + k]
                    stup = self.get_state_action_tuple(state, action)
                    if (t + k < T - 1):
                        G_t_k = (gamma ** k) * self.action_value_function[stup]
                    else:
                        gamma_pow = gamma ** (T - k - 2)
                        G_t_k = (gamma_pow * last_reward) + ((gamma_pow * gamma) * self.action_value_function[stup])
                    G_t_lambda += ((_lambda ** (k - 1)) * G_t_k)
                stup = self.get_state_action_tuple(s_t, a_t)
                self.action_value_function[stup] += alpha * (G_t_lambda - self.action_value_function[stup])
            pbar.update(1)
        pbar.close()
        
    def train_Q_backward(self, num_episodes, epsilon=0.1, decay=True, get_total_reward=False):
        pbar = tqdm(total=num_episodes)
        total_reward = []
        for i in range(num_episodes):
            
            # choose a state s
            state = self.env.initial_state()
            while(state.is_terminal):
                state = self.env.initial_state()
                
            # choose action a from epsilon greedy policy
            action = self.ep_greedy_policy(state, epsilon)
            next_action = action
            
            #initialise
            gamma = self.discount_factor
            _lambda = self._lambda
            alpha = self.alpha
            count = 2
            last_reward = 0
            
            while not state.is_terminal:
                next_state, reward = self.env.step(state, action)
                stup = self.get_state_action_tuple(state, action)
                if next_state.is_terminal:
                    delta = reward - (_lambda * self.action_value_function[stup])
                    last_reward = reward
                else:
                    next_action = self.ep_greedy_policy(next_state, (1.0 * epsilon)/ count)
                    next_tup = self.get_state_action_tuple(next_state, next_action)
                    delta = reward + ((self.action_value_function[next_tup] - self.action_value_function[stup]) * _lambda)
                self.E[stup] += 1
                self.action_value_function += (alpha * delta * self.E)
                self.E = self.E * (_lambda * gamma)
                
                state = next_state
                action = next_action
                if decay:
                    count += 1
            total_reward += [last_reward]
            pbar.update(1)
        pbar.close()
        if get_total_reward:
            return self.get_action_value_function(), total_reward
        return self.get_action_value_function()


# In[ ]:


environment = Environment()
num_episodes = 5000000
sarsa_lambda_agent = Sarsa_lambda(environment, alpha=0.1, _lambda=0.5)
Q = sarsa_lambda_agent.train_Q_backward(num_episodes, decay=True, epsilon=0.1)


# In[ ]:


Q = sarsa_lambda_agent.train_Q_backward(num_episodes, decay=True, epsilon=0.1)


# In[ ]:


plot_value_function2('Sarsa Lambda', sarsa_lambda_agent.get_value_function(), method='lambda='+str(0.5), train_steps=num_episodes, usables=[2,2,2], transpose=True)


# ### Cumulative Rewards v/s Episode Count

# In[ ]:


def get_sarsa_reward(k, decay, num_episodes=100, alpha=0.1, epsilon=0.1):
    environment = Environment()
    sarsa_agent = kSarsaControl(environment, alpha=alpha, k=k)
    _, total_reward = sarsa_agent.train_Q(num_episodes, epsilon=epsilon, decay=decay, get_total_reward=True)
    return total_reward


# In[ ]:


def cumu_reward(num_episodes, num_runs = 10):
    final_case = np.zeros([10, num_episodes])
    for run in range(num_runs):
        case1 = []
        case2 = []
        case3 = []
        case4 = []
        ks = [1, 10, 100, 1000]
        for k in ks:
            fixed_ep_reward = get_sarsa_reward(k=k, decay=False, num_episodes=num_episodes)
            var_ep_reward = get_sarsa_reward(k=k, decay=True, num_episodes=num_episodes)
            case1.append(fixed_ep_reward)
            case2.append(var_ep_reward)
        qlearning_agent = QLearningAgent(Environment(), alpha=0.1)
        _, total_reward = qlearning_agent.train_Q(num_episodes, decay=False, epsilon=0.1, get_total_reward=True)
        case3.append(total_reward)
        sarsa_lambda_agent = Sarsa_lambda(Environment(), alpha=0.1, _lambda=0.5)
        _, total_reward = sarsa_lambda_agent.train_Q_backward(num_episodes, decay=True, epsilon=0.1, get_total_reward=True)
        case4.append(total_reward)
        final_case += np.vstack((case1, case2, case3, case4))
    return ((1.0 * final_case) / num_runs)


# In[ ]:


def cumu_rewards_2(num_episodes=100, num_runs=10):
    final_case = np.zeros([10, num_episodes])
    for run in range(num_runs):
        algos = [[] for i in range(10)]
        for epi_count in range(1, num_episodes + 1):
            ks = [1, 10, 100, 1000]
            for i, k in enumerate(ks):
                fixed_ep_reward = get_sarsa_reward(k=k, decay=False, num_episodes=epi_count)
                var_ep_reward = get_sarsa_reward(k=k, decay=True, num_episodes=epi_count)
                algos[i].append(np.average(fixed_ep_reward))
                algos[i + 4].append(np.average(var_ep_reward))
            qlearning_agent = QLearningAgent(Environment(), alpha=0.1)
            _, total_reward = qlearning_agent.train_Q(epi_count, decay=False, epsilon=0.1, get_total_reward=True)
            algos[8].append(np.average(total_reward))
            sarsa_lambda_agent = Sarsa_lambda(Environment(), alpha=0.1, _lambda=0.5)
            _, total_reward = sarsa_lambda_agent.train_Q_backward(epi_count, decay=True, epsilon=0.1, get_total_reward=True)
            algos[9].append(np.average(total_reward))
        final_case += np.asarray(algos)
    return ((1.0 * final_case) / num_runs)


# In[ ]:


avg_rewards = cumu_rewards_2(num_episodes=100)


# In[ ]:


algo_names = {0:'sarsa k=1',
             1:'sarsa k=10',
             2:'sarsa k=100',
             3:'sarsa k=1000',
             4: 'sarsa k=1 decay',
             5:'sarsa k=10 decay',
             6:'sarsa k=100 decay',
             7:'sarsa k=1000 decay',
             8: 'Q learning',
             9: 'TD lambda'}


# In[ ]:


plt.figure(figsize=(21,10))
compare_list = [3, 7, 8, 9]
for i in compare_list:
    plt.plot(np.arange(100), avg_rewards[i], label=algo_names[i])
plt.legend(loc='best')
plt.show()


# ### Performance variation with Learning Rate

# In[ ]:


def generate_episode(agent, test_count=10):
    rewards = []
    for ep in range(test_count):
        episode = agent.generate_test_episode()
        T = len(episode)
        reward = episode[T - 1][2]
        rewards.append(reward)
    return np.average(rewards)


# In[ ]:


alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
num_episodes = 100000
ks = [1, 10, 100, 1000]
all_perfs = []
for alpha in alphas:
    perfs = []
    for k in ks:
        sarsa_agent = kSarsaControl(Environment(), alpha=alpha, k=k)
        Q = sarsa_agent.train_Q(num_episodes, epsilon=0.1, decay=False)
        perf = generate_episode(agent=sarsa_agent, test_count=10)
        perfs.append(perf)
    for k in ks:
        sarsa_agent = kSarsaControl(Environment(), alpha=alpha, k=k)
        Q = sarsa_agent.train_Q(num_episodes, epsilon=0.1, decay=True)
        perf = generate_episode(agent=sarsa_agent, test_count=10)
        perfs.append(perf)
    qlearning_agent = QLearningAgent(Environment(), alpha=alpha)
    Q = qlearning_agent.train_Q(num_episodes, decay=False, epsilon=0.1)
    perf = generate_episode(agent=qlearning_agent, test_count=10)
    perfs.append(perf)
    sarsa_lambda_agent = Sarsa_lambda(Environment(), alpha=0.1, _lambda=0.5)
    Q = sarsa_lambda_agent.train_Q_backward(num_episodes, decay=True, epsilon=0.1)
    perf = generate_episode(agent=sarsa_lambda_agent, test_count=10)
    perfs.append(perf)
    all_perfs.append(perfs)
all_perfs = np.asarray(all_perfs)


# In[ ]:


all_perfs.shape


# In[ ]:


algo_names = {0:'sarsa k=1',
             1:'sarsa k=10',
             2:'sarsa k=100',
             3:'sarsa k=1000',
             4: 'sarsa k=1 decay',
             5:'sarsa k=10 decay',
             6:'sarsa k=100 decay',
             7:'sarsa k=1000 decay',
             8: 'Q learning',
             9: 'TD lambda'}


# In[ ]:


plt.figure(figsize=(21,10))
compare_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in compare_list:
    plt.plot(alphas, all_perfs[:, i], label=algo_names[i], marker='o')
plt.legend(loc='best')
plt.xlabel('alpha')
plt.ylabel('average reward per episode')
plt.title('Variation of performance with learning rate')
plt.show()


# In[ ]:


plt.figure(figsize=(21,10))
compare_list = [0, 1, 2, 3, 4]
x = list(algo_names.keys())
y = list(algo_names.values())
for i in range(len(alphas)):
    plt.plot(y, all_perfs[i, :], label = alphas[i], marker='o')
plt.xticks(x, y)
plt.legend(loc='best')
plt.xlabel('agent type')
plt.ylabel('average reward per episode')
plt.title('Choosing appropriate learning rate for each agent')
plt.show()


# In[ ]:




