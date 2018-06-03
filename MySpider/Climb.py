import urllib.request
from spider import sqlHelper
import requests

class Spider(object):
    def __init__(self,url,method):
        """
            初始化url
        """
        self.url = url
        self.method = method
        pass
    def Climb(self):
        #request = urllib.request.Request(self.url)
        #response = urllib.request.urlopen(request)
        #print(response.read().decode("utf-8"))
        r = requests.get(self.url)
        #r.enconding = 'gbk'
        print(r.content.decode("gbk"))
        print('测试')
    pass
pass

if __name__ == "__main__":
    helper = sqlHelper.SQLHelper()
    lotteryno = helper.getCurentNo()#数据库中最新的数据
    spider = Spider('http://zst.aicai.com/ssq/openInfo/','POST')
    spider.Climb()