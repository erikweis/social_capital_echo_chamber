# coding: utf-8
# Echo Chamber Model w/ Social Capital Incentives
# echo_chamber_dynamics.py
# Last Update: 20200816
# by Kazutoshi Sasahara
# modified by Erik WEis

import os
import numpy as np
import networkx as nx
import pandas as pd
from agent import Agent
from social_media import SocialMedia
import analysis


class EchoChamberDynamics(object):
    def __init__(self, num_agents, num_links, epsilon, sns_seed, l, k_internal,data_dir,std):
        self.num_agents = num_agents
        self.l = l
        self.epsilon = epsilon
        self.social_media = SocialMedia(num_agents, num_links, l, sns_seed)
        self.set_agents(num_agents, epsilon,k_internal,std,self.social_media.G.out_degree())
        self.data_dir = data_dir
        self.opinion_data = []
        self.screen_diversity_data = []
        if not os.path.isdir(data_dir):
            os.makedirs(os.path.join(data_dir, 'data'))
            os.makedirs(os.path.join(data_dir, 'network_data'))

            
    def set_agents(self, num_agents, epsilon,k_internal,std,degrees):
        screen_diversity = analysis.screen_diversity([], bins=10)
        self.agents = [Agent(int(i), epsilon, screen_diversity,k_internal,std,degree) for i,degree in degrees]
        
    def total_discordant_messages(self):
        total_discordant_msgs = 0
        for a in self.agents:
            total_discordant_msgs += len(a.discordant_msgs)
        return total_discordant_msgs
   
    
    def is_stationary_state(self, G):
        
        return False

        
    def export_csv(self, data_dic, ofname):
        dir_path = os.path.join(self.data_dir, 'data')
        file_path = os.path.join(dir_path, ofname)
        pd.DataFrame(data_dic).to_csv(file_path, compression='xz')

  
    def export_gexf(self, t):
        network_dir_path = os.path.join(self.data_dir, 'network_data')
        file_path = os.path.join(network_dir_path, 'G_' + str(t).zfill(7) + '.gexf.bz2')
        cls = [float(a.opinion) for a in self.agents]
        self.social_media.set_node_colors(cls)
        nx.write_gexf(self.social_media.G, file_path)
        
    def final_exports(self, t):
        self.export_csv(self.opinion_data, 'opinions.csv.xz')
        self.export_csv(self.screen_diversity_data, 'screen_diversity.csv.xz')
        self.social_media.message_df.to_csv(os.path.join(self.data_dir + '/data', 'messages.csv.xz'))
        self.export_gexf(t)

        
    def evolve(self, t_max, mu, p, q, r,rewiring_methods,damping):
        for t in range(t_max):
            #print("t = ", t)
            
            #intervention
            if t>10000:
                for a in self.agents:
                    a.epsilon=0.4
            if t==15000:
                for a in self.agents:
                    a.epsilon=0.7
            
            self.opinion_data.append([a.opinion for a in self.agents])
            self.screen_diversity_data.append([a.screen_diversity for a in self.agents])

            # export network data
            if t % 1000 == 0:
                print(t)
                self.export_gexf(t)

            # select agent i at random
            user_id = np.random.choice(self.num_agents)

            # agent i refleshes its screen and reading it
            screen = self.social_media.show_screen(user_id)
            self.agents[user_id].evaluate_messages(screen)
            self.agents[user_id].screen_diversity = analysis.screen_diversity(screen.content.values, bins=10)

            # social influence (mu)
            unfollow_id = None
            follow_id = None
            self.agents[user_id].update_opinion(mu,self.social_media.G,damping)

            # rewiring (q)
            if np.random.random() < q:
                unfollow_id, follow_id = self.agents[user_id].decide_to_rewire(self.social_media, rewiring_methods,x)
                if unfollow_id is not None and follow_id is not None:
                    self.social_media.rewire_users(user_id, unfollow_id, follow_id)
                    u=self.agents[follow_id].decide_to_reciprocate(self.social_media,user_id,r)
                    if u:
                        self.social_media.rewire_users(follow_id, u, user_id)
                    
            # post (1-p) or repost (p) a message
            msg = self.agents[user_id].post_message(t, p)
            self.social_media.update_message_db(t, msg)
       
            # finalize and export data            
            if self.is_stationary_state(self.social_media.G):
                self.final_exports(t)
                break
            elif t >= t_max - 1:
                self.final_exports(t)
                break
        return t
                
if __name__ == '__main__':
    
    # parameters
    n_agents = 100
    n_links = 400
    sns_seed = 1
    l = 10 # screen size
    t_max = 100 # max steps
    epsilon = 0.7 # bounded confidence parameter
    mu = 0.5 # social influence strength
    p = 0.5 # repost rate
    q = 0.5 # rewiring rate
    k_internal=0.2 #internal pull force constant
    r=3 #threshhold for reciprocation
    
    following_methods = ['Random', 'Repost', 'Recommendation']
    damping=0.5 #damping of opinion influence
    std=0.6 #standard deviation of initial opinion distribution
    
    name='simulation_1'
    data_root_dir = os.path.join(os.getcwd(),'data_{}'.format(name))
    d=None
    d = EchoChamberDynamics(n_agents, n_links, epsilon, 
                            sns_seed, l, k_internal, 
                            data_root_dir,std)
    t=d.evolve(t_max, mu, p, q,r, following_methods,damping)

    

            