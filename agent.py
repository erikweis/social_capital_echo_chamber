# coding: utf-8
# Echo Chamber Model w/ Social Capital Incentives
# agent.py
# Last Update: 20200816
# by Kazutoshi Sasahara
# modified by Erik WEis

import numpy as np
from social_media import Message
from scipy.stats import truncnorm

class Agent(object):
    def __init__(self, user_id, epsilon, screen_diversity,k_internal,std,degree=1):
        
        if degree >10:
            std=std/4
        elif degree >5:
            std=std/2
        
        self.user_id = user_id
        self.base_opinion=truncnorm.rvs(-1/std,1/std,scale=std) #np.random.uniform(-1.0, 1.0)
        self.opinion=self.base_opinion
        
        if abs(self.base_opinion) > 1:
            self.k_internal=k_internal + 0.2*(abs(self.base_opinion)-0.5)
        else: 
            self.k_internal=k_internal
        
        self.epsilon = epsilon
        self.screen_diversity = screen_diversity
        self.orig_msg_ids_in_screen = []

        
    def set_orig_msg_ids_in_screen(self, screen):
        self.orig_msg_ids_in_screen = screen.orig_msg_id.values

        
    def evaluate_messages(self, screen):
        self.concordant_msgs = []
        self.discordant_msgs = []
        if len(screen) > 0:
            self.concordant_msgs = screen[abs(self.base_opinion - screen.content) < self.epsilon]
            self.discordant_msgs = screen[abs(self.base_opinion - screen.content) >= self.epsilon]
        
            
    def update_opinion(self, mu, G, damping):
        
        noise=0.0 #means the maximum noise adjustment is +/- 0.0
        max_change=1
        
        if len(self.concordant_msgs) > 0:
            force_internal=-self.k_internal*(self.opinion-self.base_opinion)
            degrees=[G.out_degree(n) for n in list(self.concordant_msgs.who_originated)]
            force_external=mu*np.average(self.concordant_msgs.content-self.opinion,weights=degrees)
            change=damping*(force_internal + force_external)+noise*truncnorm.rvs(-1,1)
            if abs(change)>max_change:
                change=max_change*np.sign(change)
            self.opinion = self.opinion + change
            
    def post_message(self, msg_id, p,base_opinion=False):
        
            
        if len(self.concordant_msgs) > 0 and np.random.random() < p:
            # repost a friend's message selected at random
            idx = np.random.choice(self.concordant_msgs.index)
            selected_msg = self.concordant_msgs.loc[idx]
            return Message(msg_id=int(msg_id), orig_msg_id=int(selected_msg.orig_msg_id),
                           who_posted=int(self.user_id), who_originated=int(selected_msg.who_originated),
                           content=selected_msg.content)
        else:
            if base_opinion==True:
                new_content=self.base_opinion
                return Message(msg_id=int(msg_id), orig_msg_id=int(msg_id),
                           who_posted=int(self.user_id), who_originated=int(self.user_id), content=new_content)
            else:    
                # post a new message
                new_content = self.opinion
                return Message(msg_id=int(msg_id), orig_msg_id=int(msg_id),
                           who_posted=int(self.user_id), who_originated=int(self.user_id), content=new_content)

        
    def decide_follow_id_at_random(self, friends,G):
        num_agents=G.number_of_nodes()
        prohibit_list = list(friends) + [self.user_id]
        option_list=[i for i in range(num_agents) if i not in prohibit_list]
        degrees=[G.out_degree(n) for n in option_list]
        degrees=[d/sum(degrees) for d in degrees]
        return int(np.random.choice(option_list,p=degrees))
    
    def decide_unfollow_id_at_random(self, discordant_messages,G):
        '''
        weight unfollowing by inverse of degree
        '''

        degrees=[G.out_degree(n) for n in discordant_messages.who_posted]
        degrees=[1/d if d>0 else 1 for d in degrees]
        unfollow_candidates = discordant_messages.who_posted.values
        return int(np.random.choice(unfollow_candidates))
    
    def decide_to_reciprocate(self,social_media,follow_id,r):
        
        '''
        unfollow lowest-degree connection
        follow random personw who follows you
        '''
        
        degree_follow_id=social_media.G.out_degree(follow_id)
        
        options=list(social_media.G.successors(self.user_id))
        unfollow_id=min(options,key=social_media.G.out_degree)
        
        ratio=degree_follow_id/np.mean(options)
        if ratio>r:
            return unfollow_id
        else: return None
        
    
    def decide_to_rewire(self, social_media, following_methods):
        unfollow_id = None
        follow_id = None
        
        if len(self.discordant_msgs) > 0:
            # decide whom to unfollow
            unfollow_id = self.decide_unfollow_id_at_random(self.discordant_msgs,social_media.G)
            # decide whom to follow
            following_method = np.random.choice(following_methods)
            friends = social_media.G.neighbors(self.user_id)

            # Repost-based selection if possible; otherwise random selection
            if following_method == 'Repost':
                friends_of_friends = list(set(self.concordant_msgs.who_originated.values) - set(friends))
                degrees=[social_media.G.out_degree(n) for n in friends_of_friends]
                degrees=[d/sum(degrees) for d in degrees]
                if len(friends_of_friends) > 0:
                    follow_id = int(np.random.choice(friends_of_friends,p=degrees))
                else:
                    follow_id = self.decide_follow_id_at_random(friends, social_media.G)

            # Recommendation-basd selection if possible; otherwise random selection
            elif following_method == 'Recommendation':
                similar_agents = social_media.recommend_similar_users(self.user_id, self.epsilon, social_media.G.number_of_nodes())
                degrees=[social_media.G.out_degree(n) for n in similar_agents]
                degrees=[d/sum(degrees) for d in degrees]
                if len(similar_agents) > 0:
                    follow_id = int(np.random.choice(similar_agents,p=degrees))
                else:
                    follow_id = self.decide_follow_id_at_random(friends, social_media.G)

            # Random selection
            else:
                follow_id = self.decide_follow_id_at_random(friends, social_media.G)
        return unfollow_id, follow_id