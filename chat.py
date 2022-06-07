from re import U
import sys

from pyparsing import match_previous_literal 

from database import Data
from jiebaSeg import JiebaSeg
from sentence import Sentence
from similarity import Similarity
from kg import KnowGraph
from syn import Syn
from quesClass import QuestionClass
from idiom import Idiom
from xiaohua import Xiaohua
import logging
""" 
各个class 说明：
Data: 从文本文件获取数据(Q&A)
JiebaSeg:分词，将一句话转化为一个列表
Sentence:保存原本数据和分词后的数据
Similarity:tf-idf模型计算相似度;将用户给定的一句话与保存数据进行相似度比较,返回最相似结果的问题的答案。
KnowGraph:知识图谱API,给定实体和属性，返回，属性的值。
Syn:从近义词辞典中查找近义词，用来查找“属性”的近义词。
QuestionClass:问题分类。
"""
class ChatAI():
    logging.basicConfig(filename='log/log.txt', level=logging.DEBUG)
    main_ans=""
    yaml_file_paths=["ai.yml","botprofile.yml","conversations.yml","emotion.yml","food.yml","gossip.yml","greetings.yml",
    "history.yml","humor.yml","literature.yml","money.yml","movies.yml","politics.yml","psychology.yml"]
    conv_file_path="conv_qa.conv"
    data_save_path = "./models/train_data.txt"  #分词后的数据保存路径
    #data_save_path="/home/lumos/.keras/datasets/spa-eng/spa.txt"
    q_data_path="./data/q_data.txt"
    a_data_path="./data/a_data.txt"
    def tf_idf_ans(self,user_q):
        sim_top2,score_top2=self.sim.get_topk(user_q,2)
        top=sim_top2[0]
        top_score=score_top2[0]
        if(top_score<0.5):
            logging.debug("没有找到符合的答案")
            return 
        logging.debug("tf_idf最佳匹配度:{top_score}".format(top_score=top_score))
        logging.debug("Q:{q}".format(q=self.question[top]))
        logging.debug("A:{a}".format(a=self.answer[top]))
        self.main_ans=self.answer[top]

    def kg_ans(self,user_q,type):
        logging.debug("知识图谱结果:")
        attribute_list,self.main_ans=self.kg.get_ans(user_q,type)
        return attribute_list

    def __init__(self) -> None:
        #未分词的问题和答复
        data=Data(self.yaml_file_paths,self.conv_file_path)
        self.question,self.answer=data.get_all_data()
        seg=JiebaSeg()
        sen=Sentence(self.question,seg)   #sen对象中包含未分词的问题，分词后的问题（词列表）
        sen.save_data(self.question,self.answer,self.data_save_path)   #保存分词后的结果
        self.sim=Similarity(sen,seg)    
        tf_idf_cur="tf_idf"
        self.sim.make_model(cur=tf_idf_cur,re_model=True) #每次修改训练数据，re_model=1,重新训练
        self.sim.save_model(cur=tf_idf_cur)
        syn=Syn()
        self.qClass=QuestionClass(seg)
        self.kg=KnowGraph(syn)
        self.idiom=Idiom()
        self.xiaohua=Xiaohua()
    def main_chat(self,user_q):
        attribute_list=[]
        type=self.qClass.get_type(user_q)
        if user_q=="":
            user_q="你是谁"
            self.tf_idf_ans(user_q)
        elif user_q=="a":
            logging.debug(attribute_list)
        elif type['type']==1 or type['type']==2 or type['type']==3:
            attribute_list=self.kg_ans(user_q,type)
        elif type['type']==4:
            self.idiom.idiom_start(start=1)
            self.main_ans="请在终端进行成语接龙"
        elif type['type']==5:
            self.main_ans=self.xiaohua.get_random()
            logging.debug(self.main_ans)
        else:
            self.tf_idf_ans(user_q)
        return self.main_ans


        




    
        