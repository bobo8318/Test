from flask.ext.script import Manager, Server
from app import app

manager = Manager(app)#Manager类将追踪所有的在命令行中调用的命令和处理过程的调用运行情况

manager.add_command("runserver", Server(host="127.0.0.1", port=8000, use_debugger=True))

@manager.command
def hello():
    print("just hello")

if  __name__ == '__main__':
    manager.run()#调用 manager.run()将启动Manger实例接收命令行中的命令。