class OpenTools():
    def __init__(self):
        pass
    def extract_tar(self, dataFile, extraDir):
        try:
            import tarfile
        except ImportError:
            raise ImportError("tarfile dont install")

        tar = tarfile.open(dataFile)
        tar.extractall(path=extraDir)
        tar.close()
        print("%s successfully extracted to %s" % (dataFile, extraDir))
        pass
    def test(self):
        print("hello world")
        pass
