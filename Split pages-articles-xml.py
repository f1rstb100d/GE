import os
import bz2
# reference: https://stackoverflow.com/questions/6184912/how-to-split-large-wikipedia-dump-xml-bz2-files-in-python
def split_xml(filename):
    if not os.path.exists("chunks"):
        os.mkdir("chunks")
    pagecount = 0
    filecount = 1
    chunkname = lambda filecount: os.path.join("chunks","chunk-"+str(filecount)+".xml.bz2")
    chunkfile = bz2.BZ2File(chunkname(filecount), 'w')
    bzfile = bz2.BZ2File(filename)
    for line in bzfile:
        chunkfile.write(line)
        if b'</page>' in line:
            pagecount += 1
            print(filecount, pagecount)
        if pagecount > 199999: #实际文件填了200000个<page>标签
            # print(chunkname()) # For Debugging
            chunkfile.write(b"</mediawiki>")
            chunkfile.close()
            pagecount = 0 # RESET pagecount
            filecount += 1 # increment filename
            chunkfile = bz2.BZ2File(chunkname(filecount), 'w')
            chunkfile.write(b"<mediawiki>\n")
    try:
        chunkfile.close()
    except:
        print('Files already close')

if __name__ == '__main__':
    split_xml('enwiki-20200220-pages-articles.xml.bz2')