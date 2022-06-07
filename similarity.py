from xml.dom.minidom import Document
import gensim 
from gensim.models.doc2vec import Doc2Vec
from gensim import corpora,models,similarities
from collections import defaultdict
import numpy as np
from sentence import Sentence

#实现方法一：使用gensim 的 doc2vec 方法
#这种方法适合用来比较长文本，经过测试，这种方法对短句效果很差
"""
# class Similarity(object):

#     def __init__(self,train_data_path) -> None:
#         data_path=train_data_path
#         train_data=[]
#         self.model=Doc2Vec(vector_size=100,window=10,min_count=5,workers=4,alpha=0.025,min_alpha=0.025,epochs=10)
#         data_file=open(data_path,"r")
#         docs=data_file.readlines()
#         data_file.close()
#         all_lines=0
#         for i,text in enumerate(docs):
#             word_list=text.split('\t')
#             document=gensim.models.doc2vec.TaggedDocument(word_list,tags=[i])
#             train_data.append(document)
#             all_lines=i
#         print(all_lines)
#         self.model.build_vocab(train_data)
#         self.model.train(train_data,total_examples=self.model.corpus_count,epochs=30)
#         self.model.save("./models/doc2vec.modle")
#     def sent2vec(self,sentence):
#         doc_vec=[]
#         for w in sentence:
#             try:
#                 doc_vec.append(self.model.wv[w])
#             except:
#                 continue
#         doc_vec=np.array(doc_vec)
#         vec_temp=doc_vec.sum(axis=0)
#         return vec_temp/np.sqrt((doc_vec**2).sum())

#     def get_sim(self,sentence):
#         sen_vector=self.sent2vec(sentence)
#         sims=self.model.docvecs.most_similar([sen_vector],topn=5)
#         return sims
"""

#方法二：使用gensim的doc2bow,similarities模块，效果较好
class Similarity(object):
    """
    sim=Similarities(sents,sen)
    sim.make_model()

    [input] class Sentences sen,class JiebaSeg seg

    [output] class Similarity
    """
    def __init__(self,sents,seg):
        self.seg=seg
        self.sen=sents
    def make_model(self,cur,re_model=False):
        """
        [input] bool re_model
        
        if re_model:从头构建模型
        
        else :  直接导入已经保存好的模型，如果没有找到，会从头构建并保存

        [output] None
        """
        cuted_sents=[]
        if re_model:   
            for sen in self.sen.get_cuted_sen():
                cuted_sents.append(sen) 
            self.dictionary=corpora.Dictionary(cuted_sents)
            cuted_sents=[]
            for sen in self.sen.get_cuted_sen():
                cuted_sents.append(sen) 
            self.dictionary=corpora.Dictionary(cuted_sents)
            temp_model=[self.dictionary.doc2bow(sen) for sen in cuted_sents]
            self.model=models.TfidfModel(temp_model)
            corpus=self.model[temp_model]
            self.index=similarities.Similarity(None,corpus,len(self.dictionary))
        else:      
            try:
                for sen in self.sen.get_cuted_sen():
                    cuted_sents.append(sen) 
                self.dictionary=corpora.Dictionary(cuted_sents)
                models_path="./models/{cur}.model".format(cur=cur)
                self.model=models.TfidfModel.load(models_path)
                index_path="./models/{cur}.index".format(cur=cur)
                self.index=similarities.Similarity.load(index_path)
                print("[DEBUG]:load(dictionary) load(model) load(index) succeed!")
            except:
                self.make_model(re_model=True)
                self.save_model(cur)
                print("[DEBUG]:load(dictionary) load(model) load(index) failed!")
                print("will self.make_model(re_model=True) ")
    def save_model(self,cur):
        dic_path="./models/{cur}.dictionary".format(cur=cur)
        self.dictionary.save(dic_path)
        models_path="./models/{cur}.model".format(cur=cur)
        self.model.save(models_path)
        index_path="./models/{cur}.index".format(cur=cur)
        self.index.save(index_path)
    def sent2vec(self,sentence):
        seg_sent=self.seg.cut(sentence)
        vec=self.dictionary.doc2bow(seg_sent)
        return self.model[vec]

    def get_topk(self,sentence,k):  
        """ 
        最相似的k个问题 
        
        [input] string sentence,int k
        
        [output] int indexs[k],double scores[k]
        """
        sentence_vec = self.sent2vec(sentence)

        sims = self.index[sentence_vec]
        sim_k = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)[:k]

        indexs = [i[0] for i in sim_k]
        scores = [i[1] for i in sim_k]
        return indexs, scores
