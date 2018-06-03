from bs4 import BeautifulSoup
import re
import urllib.parse

class HtmlParser(object):
    def parse(self, new_url, html_content):
        if(new_url is None or html_content is None):
            return
        soup = BeautifulSoup(html_content,"lxml",from_encoding="utf-8")
        new_urls = self.get_new_url(new_url,soup)
        new_data = self.get_new_data(new_url,soup)
        return new_urls,new_data
    pass

    def get_new_url(self, page_url, soup):
        new_urls = set()
        links = soup.find_all("a",href=re.compilce(r"/view/\d/.html"))
        for link in links:
            new_url = link['href']
            new_full_url = urllib.parse.urljoin(page_url,new_url);
            new_urls.add(new_full_url)
        return new_urls
    pass


    def get_new_data(self, new_url, soup):
        res_data = {}
        title_node = soup.find("dd",class_="").find("h1")
        res_data["title"] = title_node.get_text()

        summery_node = soup.find("div",class_="").find("h1")
        res_data["summery"] = summery_node.get_text()

        return res_data
    pass