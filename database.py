from site import abs_paths
from numpy import append
import yaml
import os

class Data(object):
    
    def __init__(self,yaml_file_paths,conv_file_path) -> None:
        self.yaml_file_paths=yaml_file_paths
        self.conv_file_path=conv_file_path
        self.txt_file_path="data.txt"
        
    def get_yaml_qa(self):
        """
        yaml问答数据集处理

        数据来源：
        [https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese]
        """
        q_list=[]
        a_list=[]
        file_path=os.path.abspath("./data/yaml_data")
        for file_name in self.yaml_file_paths:
            yaml_file=os.path.join(file_path,file_name)
            file=open(yaml_file,'r',encoding="utf-8")
            file_data=file.read()
            file.close()
            data=yaml.load(file_data,Loader=yaml.FullLoader)
            qa_list=data["conversations"]
            for i in qa_list:
                q_list.append(i[0])
                a_list.append(i[1])
        print("yaml_QA数据集数量:","\tQ:",len(q_list),"\tA:",len(a_list))
        return q_list,a_list

    def get_conv_qa(self):
        """
        conv文件 问答数据导入

        数据来源：
        [https://github.com/aceimnorstuvwxz/dgk_lost_conv/blob/master/results/xiaohuangji50w_nofenci.conv.zip]
        """
        qlist=[]
        alist=[]
        abs_path=os.path.abspath("./data/conv_data")
        path=os.path.join(abs_path,self.conv_file_path)
        with open(path) as file:
            count=0
            line=file.readline()
            while line!="":
                count%=3
                temp_line=str(file.readline()).split(' ')
                try:
                    if count==0:
                        line=temp_line[0]
                        qlist.append(line)
                    elif count==1:
                        line=temp_line[1]
                        alist.append(line)
                except:
                    line=""
                count=count+1
            sub=len(qlist)-len(alist)
            #print(sub)
            #print(qlist)
            if sub>0:
                while sub>0:
                    qlist.pop()
                    sub=sub-1
            elif sub<0:
                sub=-sub
                while sub>0:
                    alist.pop()
                    sub=sub-1
        print("conv_QA数据集数量:","\tQ:",len(qlist),"\tA:",len(alist))
        return qlist,alist
    def get_txt_qa(self):
        qlist=[]
        alist=[]
        abs_paths=os.path.abspath("./data/txt_data")
        path=os.path.join(abs_paths,self.txt_file_path)
        with open(path) as file:
            line=file.readline()
            while line!="":
                temp_line=str(line).split('\t')
                #print(temp_line)
                if len(temp_line)==2:
                    qlist.append(temp_line[0])
                    alist.append(temp_line[1].strip('\n'))
                line=file.readline()
        sub=len(qlist)-len(alist)
        #print(sub)
        #print(qlist)
        if sub>0:
            while sub>0:
                qlist.pop()
                sub=sub-1
        elif sub<0:
            sub=-sub
            while sub>0:
                alist.pop()
                sub=sub-1
        # for i in range(len(alist)):
        #     print("['{q}','{a}'\n".format(q=qlist[i],a=alist[i]))
        print("txt_data数据集数量:","\tQ:",len(qlist),"\tA:",len(alist))
        return qlist,alist
    def get_all_data(self):
        question=[]
        answer=[]
        #q_yaml,a_yaml=self.get_yaml_qa()    
        q_yaml=[]
        a_yaml=[]
        q_conv,a_conv=self.get_conv_qa()
        q_txt,a_txt=self.get_txt_qa()
        question=q_yaml+q_conv+q_txt
        answer=a_yaml+a_conv+a_txt
        print("all_data数据集数量:","\tQ:",len(question),"\tA:",len(answer))
        return question,answer

