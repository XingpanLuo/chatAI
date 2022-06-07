import random
import logging
class Idiom(object):
    main_ans=""
    def idiom_exists(self,x):
        #判断是否为成语的函数，参数为字符串，判断该字符串是否在成语库中
        with open('./data/idiom.txt','r',encoding='utf-8') as f:
            for i in set(f.readlines()):
                if x == i.strip():
                    return True
            return False

    def idiom_test(self,idiom1, idiom2):
        #判断两个成语是否达成接龙条件
        if idiom2[0] != idiom1[-1] or len(idiom2) != 4:
            return False
        return True

    def idiom_select(self,x):
        if x == None:
            with open('./data/idiom.txt','r',encoding='utf-8') as f:
                return random.choice(f.readlines())[:-1]
        else:
            with open('./data/idiom.txt','r',encoding='utf-8') as f:
                base = f.readlines()
                random.shuffle(base)
                for i in base:
                    if i[:-1] == x or len(i) != 5:
                        continue
                    if i[0] == x[-1]:
                        return i[:-1]
            return None

    def idiom_start(self,start = 0):
        #start参数表示先后手，0表示电脑先手，1表示玩家先手；返回值代表游戏结果，为0表示玩家失败，为1代表玩家胜利
        memory = set()  #记忆集合，用于判断成语是否被重复使用
        if start == 0:
            while True:
                t = self.idiom_select(None)
                if self.idiom_select(t) != None and len(t) == 4:
                    break
            logging.debug(t)
            print(t)
            self.main_ans=t 
        else:
            p = input("请输入成语:")
            if p.strip() == '':
                logging.debug("好吧，让我先开始")
                print("好吧，让我先开始")
                self.idiom_start(0)
            if self.idiom_exists(p) == False:
                logging.debug("游戏结束！该成语不存在")
                print("游戏结束！该成语不存在")
                self.main_ans="游戏结束！该成语不存在"
                return self.main_ans
            memory.add(p)
            cycle_flag = 0  #控制while True循环次数
            while True:
                t = self.idiom_select(p)
                cycle_flag += 1
                if t not in memory:
                    break
                if cycle_flag == 10:
                    t = None
                    break
            if t == None:
                logging.debug("恭喜你，你赢了！")
                print("恭喜你，你赢了！")
                self.main_ans="恭喜你，你赢了！"
                return self.main_ans
            else:
                logging.debug(t)
                memory.add(t)        
        while True:
            p = input("请输入成语:")
            if p.strip()=='':
                # logging.debug("t:",t)
                p = self.idiom_select(t)
                if p!=None:
                    self.main_ans="好吧，我帮你一次:{p}".format(p=p)
                    logging.debug("好吧，我帮你一次:{p}".format(p=p))
                    print("好吧，我帮你一次:{p}".format(p=p))
                else:
                    p='q'
                    self.main_ans=""
            if p.strip() == 'q':
                logging.debug("游戏结束！你输了")
                print("游戏结束！你输了")
                self.main_ans="游戏结束！你输了"
                return self.main_ans
            if self.idiom_exists(p) == False:
                logging.debug("游戏结束！该成语{p}不存在".format(p=p))
                print("游戏结束！该成语{p}不存在".format(p=p))
                self.main_ans="游戏结束！该成语不存在"
                return self.main_ans
            if p in memory:
                logging.debug("游戏结束！该成语已被使用过")
                print("游戏结束！该成语已被使用过")
                self.main_ans="游戏结束！该成语不存在"
                return self.main_ans
            if self.idiom_test(t, p) == False:
                logging.debug("游戏结束！你未遵守游戏规则")
                print("游戏结束！你未遵守游戏规则")
                self.main_ans="游戏结束！你未遵守游戏规则"
                return self.main_ans
            memory.add(p)
            cycle_flag = 0
            while True:
                try:
                    t = self.idiom_select(p)
                    cycle_flag += 1
                    if t not in memory:
                        break
                    if cycle_flag == 10:
                        t = None
                        break
                except:
                    break
            if t == None:
                logging.debug("恭喜你，你赢了！")
                print("恭喜你，你赢了！")
                self.main_ans="恭喜你，你赢了"
                return self.main_ans
            else:
                print(t)
                logging.debug(t)
                memory.add(t)

if __name__ == '__main__':
    idiom=Idiom()
    idiom.idiom_start(start=1)