from html import entities
import imp
from inspect import Attribute
from time import sleep
import requests
from weather import Weather
import logging
class KnowGraph(object):
	main_ans=""
	def __init__(self,syn) -> None:
		self.syn=syn
	def mention2entity(self,mention):
		'''
		* mention2entity - 提及->实体
		'''
		url = 'https://api.ownthink.com/kg/ambiguous?mention={mention}'.format(mention=mention)      # 知识图谱API，歧义关系
		sess = requests.get(url) # 请求
		text = sess.text # 获取返回的数据
		entitys = eval(text) # 转为字典类型
		return entitys
		
	def entity2knowledge(self,entity):
		'''
		* 根据实体获取实体知识
		'''
		url = 'https://api.ownthink.com/kg/knowledge?entity={entity}'.format(entity=entity)      # 知识图谱API，实体知识
		sess = requests.get(url) # 请求
		text = sess.text # 获取返回的数据
		knowledge = eval(text) # 转为字典类型
		#knowledge=knowledge['data']['value']
		return knowledge
	def entity_attribute2value(self,entity, attribute):
		'''
		* 根据实体、属性获取属性值
		'''
		url = 'https://api.ownthink.com/kg/eav?entity={entity}&attribute={attribute}'.format(entity=entity, attribute=attribute)      # 知识图谱API，属性值
		sess = requests.get(url) # 请求
		text = sess.text # 获取返回的数据
		values = eval(text) # 转为字典类型
		#logging.debug(values)
		return values
	def type_one(self,type):
		sim_attribute=[]
		entity=type['entity']
		attribute=type['attribute']
		knowledge=self.entity2knowledge(entity)
		knowledge=knowledge['data']['avp']
		kg_ans=self.entity_attribute2value(entity,attribute)
		flag=False
		for i in knowledge:
			sim_attribute.append(i[0])
		if 'value' in kg_ans['data']:
			logging.debug("找到了{entity}的{attribute}是".format(entity=entity,attribute=attribute))
			logging.debug(kg_ans['data']['value'][0])
			self.main_ans=kg_ans['data']['value'][0]
			logging.debug(10*"*")
			sleep(0.5)
			flag=True
		else:
			near_attribute=self.syn.nearby(attribute,5)
			for near in near_attribute:
				kg_ans=self.entity_attribute2value(entity,near)
				if 'value' in kg_ans['data']:
					logging.debug("找不到{entity}的{attribute}属性".format(entity=entity,attribute=attribute))
					logging.debug("但是找到了{entity}的{attribute}属性".format(entity=entity,attribute=near))
					logging.debug(kg_ans['data']['value'][0])
					self.main_ans=kg_ans['data']['value'][0]
					logging.debug(10*"*")
					flag=True
					break 
		if(flag==False):
			most_attribute=self.syn.most_sim(attribute,sim_attribute)
			try:
				kg_ans=self.entity_attribute2value(entity,most_attribute)
				#logging.debug(most_attribute)
				logging.debug("找不到{entity}的{attribute}属性".format(entity=entity,attribute=attribute))
				logging.debug("但是找到了{entity}的{attribute}属性".format(entity=entity,attribute=most_attribute))
				logging.debug(kg_ans['data']['value'][0])
				self.main_ans=kg_ans['data']['value'][0]
				sleep(0.5)
				logging.debug(10*"*")
			except:
				self.main_ans="抱歉，根据知识图谱查找{entity}的{attribute}失败".format(entity=entity,attribute=attribute)
				logging.debug("抱歉，根据知识图谱查找{entity}的{attribute}失败".format(entity=entity,attribute=attribute))
		return sim_attribute
	def type_two(self,type):
		entity=type['entity']
		attribute=type['attribute']
		sim_attribute=[]
		data=self.entity2knowledge(entity)
		try:
			desc=data['data']['desc']
			self.main_ans=desc 
			logging.debug(desc)
		except:
			desc="没有找到合适的答案"
			self.main_ans=desc 
			try:
				sim_a=self.mention2entity(entity)
				sim_attribute.append(sim_a[0][0])
				sim_attribute.append(sim_a[1][0])
				sim_attribute.append(sim_a[2][0])
			except:
				pass 
			logging.debug(sim_attribute)
			logging.debug("没有找到"+entity+"的信息")
		#logging.debug(data)
		return sim_attribute,desc
	
	def type_three(self,type):
		entity=type['entity']
		attribute=type['attribute']
		weather=Weather()
		logging.debug(entity)
		wea=weather.get_weather(entity,attribute)
		self.main_ans=wea 
		logging.debug(wea)
	def get_ans(self,sen,type):
		sim_attribute=[]
		if type['type']==1:
			sim_attribute=self.type_one(type)
		elif type['type']==2:
			self.type_two(type)
		elif type['type']==3:
			self.type_three(type)
		else:
			self.main_ans="我还不理解你的意思^-^~"
			logging.debug("我还不理解你的意思^-^~")
		return sim_attribute,self.main_ans

