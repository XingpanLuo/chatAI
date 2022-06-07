import re
import thulac
class QuestionClass(object):
    def __init__(self,seg) -> None:
        self.thu=thulac.thulac(user_dict="./userdict/userdict.txt",seg_only=False)
        self.seg=seg
        # self.when=['时候','时间','何时','什么时候','日期','年','月','日','多久']
        # self.where=['在哪儿','位置','地点','地址','哪儿']
        # self.who=['谁','叫什么','是谁','名称','姓名']
        # self.what=['是什么','定义','是多少','有多高']
        self.nPos=['n','np','ns','ni','nz','uw']
        self.uPos=['u']
        self.vPos=['v']
        self.aPos=['a','d']
    
    def pos_cut(self,sen):
        words=[]
        flags=[]
        thupos=self.thu.cut(sen)
        for i in thupos:
            words.append(i[0])
            flags.append(i[1])
        for i in range(len(flags)):
            if flags[i] in self.nPos:
                flags[i]='n'
            elif flags[i] in self.uPos:
                flags[i]='u'
            elif flags[i] in self.vPos:
                flags[i]='v'
            elif flags[i] in self.aPos:
                flags[i]='a'
            else:
                flags[i]='x'
        return words,flags
    def get_type(self,sen):
        words,flags=self.pos_cut(sen)
        str=""
        for i in flags:
            str+=i 
        nun=re.search('nun',str)#美国的总统
        nn=re.search('nn',str)#中国首都
        onen=re.search('n',str)#姚明是谁
        xn=re.search('xn',str)#后天天气
        nxn=re.search('nxn',str)#合肥昨天天气
        type=0
        entity="None"
        attribute="None"
        if '成语' in words:
            type=4
        elif '笑话' in words:
            type=5
        elif nun!=None:
            type=1
            span=nun.span()
            entity=words[span[0]]
            attribute=words[span[1]-1]
        elif nn!=None:
            type=1
            span=nn.span()
            entity=words[span[0]]
            attribute=words[span[1]-1]
        elif nxn!=None:
            span=xn.span()
            if words[span[1]-1]=='天气':
                type=3
            entity=words[span[0]-1]
            if ('今' in words )|('今天' in words):
                attribute=0
            elif ('明' in words )|('明天' in words):
                attribute=1
            elif ('后' in words )|('后天' in words):
                attribute=2
            else:
                attribute=0
        elif xn!=None:
            span=xn.span()
            if words[span[1]-1]=='天气':
                type=3
            entity='合肥'
            if ('今' in words )|('今天' in words):
                attribute=0
            elif ('明' in words )|('明天' in words):
                attribute=1
            elif ('后' in words )|('后天' in words):
                attribute=2
            else:
                attribute=0
        elif onen!=None:
            type=2
            span=onen.span()
            entity=words[span[0]]
            attribute=""
        else:
            span=(0,0)
        questionClass={"type":type,"entity":entity,"attribute":attribute}
        return questionClass


# if __name__ == '__main__':
#     seg=JiebaSeg()
#     qclass=QuestionClass(seg)
#     que=[]
#     que.append('北京明天天气怎么样')
#     que.append('后天天气怎么样')
#     que.append('北京的首都')
#     que.append('美国总统')
#     que.append('姚明是谁')
#     for q in que:
#         qclass.get_type(q)

    # while(que!='q'):
    #     qclass.get_type(que)
    #     que=input()
