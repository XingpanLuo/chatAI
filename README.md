# 聊天机器人 文档

## 支持服务

  * 基于if_idf的相似度匹配聊天
  * 基于知识图谱的聊天机器人
  * 支持查询天气
  * 成语接龙小游戏
  * 支持讲笑话   
  * 基础图形界面(tkinter-GUI) 

## 目录结构

-chatAI
  -__pychache__ : python 自动生成的中间文件,删除不影响
  -data : 各类数据
    -conv_data,txt_data,yaml_data : 从三个途径获得的问答语料库,分别进行处理
    -a_data.txt,q_data.txt : 上面三个问答数据清洗后分词结果
    -idiom.txt : 成语辞典,存放几千个成语
    -xiaohua.txt : 存放爬取并清洗后的笑话合集
  -log : 日志文件
  -models : 存放模型,主要是tf_idf中保存的模型,如果已经有模型了,直接导入就可以不用每次都重新生成
  -stopwordList: 结巴分词的停用词
  -THULAC-Python : 使用thulac获取词性,例如中国是名词(n),“回答”是动词(v)
  -unused : 没完成的功能/函数
  -userdict : 用户词典,比如将“中国科学技术大学”加入词典,这个词就不会被分割了
  -chat.py : 各个类的初始化,核心模块
  -database.py : 对./data中的文件进行清洗
  -gui.py : 生成图形界面
  -idiom.py : 成语接龙游戏
  -jieba.py : 结巴分词的封装,对一句话进行分词
  -kg.py : 知识图谱模块
  -main.py : 启动模块,其实就是把ChatAI和GUI结合
  -quesClass.py : 问题分类,比如“你是谁”分为0类,则使用if_idf得到结果;“中国的首都”分为3类,使用知识图谱得到结果
  -sentence.py : 分词并保存原本数据和分词后的数据
  -similarity.py : tf-idf模型计算相似度;将用户给定的一句话与保存数据进行相似度比较,返回最相似结果的问题的答案。
  -syn.py : 同样是计算相似度,但这个是计算两个词的相似度,需要加载一个近义词词典
  -weather.py : 获取某地某时的天气,实际上获取了一周的天气预报数据,但聊天模块只支持查询今天和明天。。。奇怪的是结巴分词“明天”和“后天”的词性不同,故不支持后天天气查询
  -xiaohua.py : 讲笑话模块,爬取笑话数据,清洗后保存,需要时随机返回

## 整体框架

![流程图](https://cdn.jsdelivr.net/gh/XingpanLuo/PictureBed/BlogImag202206071324423.jpg)

## 功能模块测试

### 知识图谱：

![kg](https://cdn.jsdelivr.net/gh/XingpanLuo/PictureBed/BlogImag202206071324234.png)

说明：

* 查询姚明的相关信息，得到合理输出
* 询问中国首都，输出正确结果
* 询问美国总统，日志显示查询失败，”找不到美国的总统属性，但找到了美国的首都属性“
* 查询中国科学技术大学的英文名，输出结果正确
* 查询南京大学的外文名，输出结果错误

### 成语接龙

![idiom1](https://cdn.jsdelivr.net/gh/XingpanLuo/PictureBed/BlogImag202206071324609.png)

![idiom2](https://cdn.jsdelivr.net/gh/XingpanLuo/PictureBed/BlogImag202206071325749.png)

说明：

* 成语接龙，用户输
* 成语接龙，用户赢

### 查询天气

![weather](https://cdn.jsdelivr.net/gh/XingpanLuo/PictureBed/BlogImag202206071324168.png)

说明：

* 查询明天天气，默认地点为合肥，输出正确
* 未指明时间和地点，输出”天气“的解释
* 查询今天天气，默认地点为合肥，输出正确
* 查询北京今天天气和上海明天天气，输出均正确

### 笑话

![xiaohua](https://cdn.jsdelivr.net/gh/XingpanLuo/PictureBed/BlogImag202206071324810.png)

说明：

* 用户输入含“笑话”关键词，随机返回笑话

### tf-idf

tf-idf相似度匹配算法

![tf_idf](https://cdn.jsdelivr.net/gh/XingpanLuo/PictureBed/BlogImag202206071324217.png)

说明：

* “你是谁”，日志显示匹配度为0.92 
* “你几岁了”，日志显示匹配度为0.59
* “你喜欢什么颜色”，日志显示匹配度为0.96

## 引用说明

### 数据引用

* ./data/conv_data/conv_qa.conv :    [https://github.com/aceimnorstuvwxz/dgk_lost_conv/blob/master/results/xiaohuangji50w_nofenci.conv.zip]
* ./data/txt_data/data.txt:[chatAI/data/txt_data at main · XingpanLuo/chatAI (github.com)](https://github.com/XingpanLuo/chatAI/tree/main/data/txt_data)
* ./data/yaml_data/: [https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese]
* ./stopwordList/stopword.txt : https://blog.csdn.net/woshishui68892/article/details/108203121 自己做了修改
* ./userdict/userdict.txt : 自定义
* ./data/xiaohua.txt : http://xiaohua.zol.com.cn/lengxiaohua/ 爬取并清洗
* ./data/idiom.txt :[idiom.txt_免费高速下载|百度网盘-分享无限制 (baidu.com)](https://pan.baidu.com/s/1dFFyHQ5) [提取码：6eiv]

### 开源项目引用

* 结巴分词：https://github.com/fxsjy/jieba

* gensim : https://gensim.apachecn.org/#/

* thulac 词性API : 其实是分词API，结巴分词也可以显示词性，但感觉thulac 效果更好 

  [thunlp/THULAC-Python: An Efficient Lexical Analyzer for Chinese (github.com)](https://github.com/thunlp/THULAC-Python)

* synonyms 近义词词典：[chatopera/Synonyms: 中文近义词：聊天机器人，智能问答工具包 (github.com)](https://github.com/chatopera/Synonyms)

* 思知 知识图谱API : [知识图谱 - 思知（OwnThink）](https://www.ownthink.com/docs/kg/)

* 免费天气API（一周天气） ：[免费七日天气接口API 未来一周天气预报api (yiketianqi.com)](https://www.yiketianqi.com/index/doc?version=week)

### 参考引用：

* gensim 使用例子：https://www.cnblogs.com/softmax/p/9042397.html
* gensim简单qna系统例子：https://zhuanlan.zhihu.com/p/163645532
* 结巴分词使用例子：官网(github) [jieba/test at master · fxsjy/jieba (github.com)](https://github.com/fxsjy/jieba/tree/master/test)
* 成语接龙参考：https://zhuanlan.zhihu.com/p/26951012
* 天气查询接口参考：官网示例：[免费七日天气接口API 未来一周天气预报api (yiketianqi.com)](https://www.yiketianqi.com/index/doc?version=week)
* 知识图谱使用：官网示例 [知识图谱 - 思知（OwnThink）](https://www.ownthink.com/docs/kg/)
* tkinter GUI使用 [菜鸟教程](https://www.runoob.com/python/python-gui-tkinter.html)  , http://c.biancheng.net/tkinter/

