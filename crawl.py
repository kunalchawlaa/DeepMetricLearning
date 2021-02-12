import requests, os, csv, sys, time
from requests.exceptions import ConnectionError

dirname = "perfect500"
startindex = sys.argv[1]

def saveProductImages():
    ifile = open('metadata_public.csv', 'rb')
    lines = csv.reader(ifile)
    next(lines) # skipping headers
    for x in xrange(int(startindex)):
        next(lines)
    for line in lines:
        url = line[1]
        extention = 'jpg'
        if(url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.png'))):
            extention = url.split('.')[-1]
            
        imageName = os.path.join(dirname, line[0] + '.' + extention)
        if(os.path.isfile(imageName)):
            print 'File exists: '+imageName
            continue
        else:
            print 'Downloading: '+imageName

        with open(imageName, 'wb') as handle:
            response = requests.get(url, stream=True, verify="/home/kunal/Documents/SRnD+Web+Proxy.crt", timeout=10)
            if not response.ok:
                print response
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)
        statinfo = os.stat(imageName)
        print 'Size: ' + str(statinfo.st_size)

# Download all product images
if __name__ == "__main__":
        if(not os.path.exists(dirname)):
            os.mkdir(dirname)
        saveProductImages()


