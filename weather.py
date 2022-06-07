import requests

class Weather(object):
    city='合肥'
    def __init__(self,city='合肥') -> None:
        self.city=city
        self.req_weather(city)
    def req_weather(self,city):
        self.city=city 
        r=requests.get('https://www.yiketianqi.com/free/week?appid=69629164&appsecret=ig9PITOe&unescape=1&city='+self.city)
        r.encoding = 'utf-8'
        self.weather=r.json()['data']

    def get_weather(self,city='合肥',index=0):
        if city!=self.city:
            self.req_weather(city)
        if index>6:
            print("只支持查询一周内的天气情况,将查询当日天气")
            index=0
        r=self.weather[index]
        rstr=self.city+'    '+r['date']+'     '+r['wea']+'   '+r['tem_night']+' - '+r['tem_day']+' 度'
        return rstr 

if __name__ == '__main__':
    wea=Weather('合肥')
    print(wea.get_weather(index=0))
