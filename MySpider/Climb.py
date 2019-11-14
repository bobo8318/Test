#from spider import sqlHelper
import requests
from spider import html_parser

class Spider(object):
    def __init__(self,url,method,data):
        """
            初始化url
        """
        self.url = url
        self.method = method
        self.data = data
        pass
    def Climb(self):
        #request = urllib.request.Request(self.url)
        #response = urllib.request.urlopen(request)
        #print(response.read().decode("utf-8"))
        r = requests.get(self.url,data = self.data)
        #r.enconding = 'gbk'
        #print(r.content.decode("utf-8"))
        print("climb done start parse...")
        parser = html_parser.HtmlParser()
        parser.get_cp_data(r.content.decode("utf-8"))
    pass
pass

if __name__ == "__main__":
    #helper = sqlHelper.SQLHelper()
    #lotteryno = helper.getCurentNo()#数据库中最新的数据
    data = {'maxsize':100,'sortTag':'down','pageSize':100,'currentPage':1,'pageInfo':'openInfo','gameIndex':101}
    spider = Spider('http://zst.aicai.com/ssq/openInfo/','post',data)
    spider.Climb()