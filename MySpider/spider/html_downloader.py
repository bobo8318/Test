import urllib.request
import urllib.parse

class HtmlDownloader(object):
    def dowload(self, new_url):
        if(new_url is None):
            return None
        req = urllib.request.Request(url=new_url)
        response = urllib.request.urlopen(req)
        if(response.getcode()!=200):
            return None
        return response.read()
    pass
pass