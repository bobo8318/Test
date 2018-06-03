from spider import html_downloader
from spider import html_outputer
from spider import html_parser
from spider import url_manager


class SpiderMain(object):
    def __init__(self):
        self.url = url_manager.UrlManager()
        self.downloader = html_downloader.HtmlDownloader()
        self.parser = html_parser.HtmlParser()
        self.outputer = html_outputer.HtmlOutputer()
    pass

    def craw(self, root_url):
        count = 1
        self.url.addNewUrl(root_url)
        while(self.url.hasNewUrl()):
            try:
                new_url = self.url.get_new_url();
                html_content = self.downloader.dowload(new_url)
                new_urls,new_data = self.parser.parse(new_url,html_content)
                self.url.addNewUrls(new_urls)
                self.outputer.collect_data(new_data)
                print("craw %d:%s"%(count,new_url))
                if(count == 1000):
                    break
                count += 1
            except:
                print("error url %s"%(new_url))
            pass
        pass

        self.outputer.output_html()
    pass

if __name__ == '__main__':
    root_url = ""
    obj_spider = SpiderMain()
    obj_spider.craw(root_url)

pass