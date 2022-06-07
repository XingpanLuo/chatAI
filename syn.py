import synonyms
import sys
class Syn(object):
    def __init__(self) -> None:
        self.syn=synonyms

    def compare(self,sen1,sen2,is_seg=False):
        savedStdout = sys.stdout  #保存标准输出流
        file = open('./log/log.txt', 'w+')
        sys.stdout = file  #标准输出重定向至文件
        score=synonyms.compare(sen1,sen2,is_seg)
        sys.stdout = savedStdout 
        return score
    
    def most_sim(self,sen,sens):
        index=0
        max_score=0
        for i in range(len(sens)):
            score=self.compare(sen,sens[i],False)
            #print(score)
            if score>max_score:
                max_score=score 
                index=i
        return sens[index]
    def nearby(self,sen,count=5):
        near=synonyms.nearby(sen,count)
        return near[0] 

# if __name__=='__main__':
#     syn=Syn()
#     sens=['中文名', '外文名', '简称', '创办时间', '类别', '类型', '属性', 
#     '主管部门', '现任领导', '专职院士', '本科专业', '硕士点', '博士点', '博士后', 
#     '国家重点学科', '院系设置', '校训', '校歌', '校庆日', '地址', '院校代码', 
#     '主要奖项', '知名校友', '所属地区', '国家实验室', '国重实验室', '教育部实验室', 
#     '创办人', '学校类型', '学校属性', '现任校长', '主要院系', '院士', '长江学者', 
#     '博士后流动站', '发展目标', '办学性质', '学校类别', '学校特色', '专职院士数']
#     sen='建校时间'
#     ans=syn.most_sim(sen,sens)
#     print(syn.nearby("何时",3))
#     print(ans)
        

