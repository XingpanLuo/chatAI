import jieba
import jieba.analyse
import jieba.posseg as pseg 
class JiebaSeg(object):
    """
    seg=JiebaSeg()
    seg.cut("我是中国科学技术大学的学生")
    """
    stopword_path="./stopwordList/stopword.txt"
    userdict_path="./userdict/userdict.txt"
    def __init__(self) -> None:
        self.stopword=set() #停用词，一个无序集合
        self.get_stopword()
        jieba.load_userdict(self.userdict_path)
        

    def get_stopword(self):
        """加载停用词"""
        file=open(self.stopword_path,'r',encoding="utf-8")
        file_data=file.readlines()
        file.close()
        for line in file_data:
            line=line.strip('\r\n')
            self.stopword.add(line)

    def cut(self,sentence,mv_stopwords=True):
        """
        if mv_stopwwords:分词，并去除停用词

        else:分词，不去除停用词
        """
        seg_list=jieba.cut(sentence)    #分词
        seg_result=[]
        if mv_stopwords:
            for seg in seg_list:    #去除停用词
                    if seg not in self.stopword:
                        seg_result.append(seg)
        else:
            seg_result=seg_list
        return seg_result 
    
    def analyse(self,sens,topK,withWeight=False,allowPOS=()):
        top=jieba.analyse.extract_tags(sens,topK,withWeight,allowPOS)
        return top

    def pos_cut(self,sen):
        ans=pseg.cut(sen)
        words=[]
        flags=[]
        for word,flag in ans:
            words.append(word)
            flags.append(flag)
        #print(words,flags)
        return words,flags

