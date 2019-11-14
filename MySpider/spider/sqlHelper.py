import pymysql

class SQLHelper(object):
   def  __init__(self):
        self.host = 'localhost'
        self.port = 3307
        self.username = 'root'
        self.password = 'usbw'
        self.database = 'openui'
        
   def getCurentNo(self):
        #db = pymysql.connect(self.host+":"+self.port,self.username,self.password,self.database)
        db = pymysql.connect(host=self.host,
                          port=self.port,
                          user=self.username,
                          passwd=self.password,
                          db=self.database,
                          charset='utf8')
        cursor = db.cursor()
        cursor.execute("select * from lottery order by lotteryno desc limit 0,1 ")
        data = cursor.fetchone()
        db.close()
        return data
    
if __name__ == '__main__':
   sql = SQLHelper()
   print(sql.getCurentNo())
pass