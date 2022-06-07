from matplotlib import use
from chat import ChatAI
import os 
from gui import *
import logging
if __name__=='__main__':
    chat=ChatAI()   #聊天机器人核心
    logging.basicConfig(filename='log/log.txt', filemode='w',level=logging.DEBUG)
    f=open('log/gui_log.txt','w')
    f.truncate()
    f.close()
    f=open('log/log.txt','w')
    f.truncate()
    f.close()
    gui(chat)
    