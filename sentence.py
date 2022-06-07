# 句子和分词后的词列表

class Sentence(object):
    origin_sens=[]
    cuted_sens=[]
    def __init__(self,sentences,seg,is_seg=True) -> None:
        self.origin_sens=sentences
        self.cuted_sens=[]
        self.seg=seg 
        for sentence in sentences:
            if is_seg:
                seg_sen=self.seg.cut(sentence)
            else:
                temp=[]
                temp.append(sentence)
                seg_sen=temp
            self.cuted_sens.append(seg_sen)

    def save_data(self,question,answer,data_save_path):
        self.data_save_path=data_save_path
        file=open(data_save_path,"a")
        file.seek(0)
        file.truncate()
        que=[]
        ans=[]
        if len(question)<len(answer):
            min_len=len(question)
        else:
            min_len=len(answer)
        for i in range(min_len):
            seg_que=self.seg.cut(question[i].strip('\n'))
            #logging.debug(answer)
            seg_ans=self.seg.cut(answer[i].strip('\n'))
            if len(seg_que)!=0:
                for s in seg_que:
                    file.write(s+" ")
                file.write("	")
                for s in seg_ans:
                    file.write(s+" ")
                file.write("\n")
        file.close()
    def get_origin_sen(self):   #原来的句子
        return self.origin_sens

    def get_cuted_sen(self):    #分词后的句子（列表）
        return self.cuted_sens

    def get_data_path(self):    #分词后数据保存的文件
        return self.data_save_path