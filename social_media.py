# coding: utf-8
# Echo Chamber Model w/ Social Capital Incentives
# social_media.py
# Last Update: 20200816
# by Kazutoshi Sasahara
# modified by Erik WEis

import numpy as np
import networkx as nx
import pandas as pd
import random

class Message(object):
    def __init__(self, msg_id, orig_msg_id, who_posted, who_originated, content):
        self.msg_id = msg_id
        self.orig_msg_id = orig_msg_id
        self.who_posted = who_posted
        self.who_originated = who_originated
        self.content = content


    def to_dict(self):
        return {'msg_id': self.msg_id, 'orig_msg_id':self.orig_msg_id,
                'who_posted':self.who_posted, 'who_originated':self.who_originated,
                'content':self.content,}



class SocialMedia(object):
    def __init__(self, num_agents, num_links, l, sns_seed):
        self.num_agents = num_agents
        random_state = np.random.RandomState(sns_seed)
        
        #create power-law degree sequence
        degree_cap=50
        power_sequence=[int(i) for i in nx.utils.powerlaw_sequence(num_agents)]
        p=np.array(power_sequence)/sum(power_sequence)
        out_degree=[1]*num_agents
        in_degree=[1]*num_agents
        for _ in range(num_links-num_agents):
            r_in=np.random.choice(num_agents)
            added_in=False
            while not added_in:
                if in_degree[r_in]<degree_cap:
                    in_degree[r_in]+=1
                    added_in=True
                else: r_in=np.random.choice(num_agents)
            
            r_out=np.random.choice(num_agents,p=p)
            added_out=False
            while not added_out:
                if out_degree[r_out]<degree_cap:
                    out_degree[r_out]+=1
                    added_out=True
                else: r_out=np.random.choice(num_agents,p=p)
        
        #create graph
        G=nx.directed_configuration_model(in_degree,out_degree,seed=random_state)
        G=nx.DiGraph(G)
        self.G=G
        self.modify_configuration_graph(G)
        
        #create message data structure
        self.message_dic = {}
        self.message_df = pd.DataFrame(columns=['msg_id', 'orig_msg_id', 'who_posted', 'who_originated', 'content'])
        self.screen_size = l
    
    def modify_configuration_graph(self,G):
        edges_to_remove=list(nx.selfloop_edges(G))
        for i,j in edges_to_remove:
            G.remove_edge(i,j)
            choice_list=[node for node in G.nodes if node not in G.successors(i)]
            G.add_edge(i,random.choice(choice_list))
        self.G=G

    def modify_random_graph(self):
        for no_outdegree_node in [k for k, v in list(self.G.out_degree()) if v == 0]:
            target_node = np.random.choice([k for k, v in list(self.G.out_degree()) if v >= 2])
            i = np.random.choice(len(self.G.edges(target_node)))
            target_edge = list(self.G.edges(target_node))[i]
            self.G.remove_edge(target_edge[0], target_edge[1])
            self.G.add_edge(no_outdegree_node, target_edge[1])


    def set_node_colors(self, node_colors):
        for i, c in enumerate(node_colors):
            self.G.nodes[i]['color'] = c


    def show_screen(self, user_id):
        friends = self.G.neighbors(user_id)
        friend_message_df = self.message_df[self.message_df['who_posted'].isin(friends)]
        friend_message_df = friend_message_df[friend_message_df['who_originated'] != user_id]
        return friend_message_df.tail(self.screen_size)


    def update_message_db(self, t, msg):
        self.message_df = self.message_df.append(msg.to_dict(), ignore_index=True).tail(self.num_agents)
        

    def recommend_similar_users(self, user_id, epsilon, num_agents):
        similar_users = []
        my_message_df = self.message_df[self.message_df.who_originated == user_id].tail(1)

        if len(my_message_df) > 0:
            last_message = my_message_df.content.values[0]
            friends = self.G.neighbors(user_id)
            similar_messages_df = self.message_df[self.message_df.who_originated != user_id].tail(num_agents)
            similar_messages_df = similar_messages_df[abs(last_message - similar_messages_df.content) < epsilon]
            if len(similar_messages_df) > 0:
                similar_users = [u for u in similar_messages_df.who_originated.values if u not in friends]

        return similar_users


    def rewire_users(self, user_id, unfollow_id, follow_id):
        self.G.remove_edge(user_id, unfollow_id)
        self.G.add_edge(user_id, follow_id)