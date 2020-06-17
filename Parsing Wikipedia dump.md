Merge Wikipedia's page and pagelinks sql dump

Download [page sql dump](https://dumps.wikimedia.org/enwiki/20200220/enwiki-20200220-page.sql.gz), and extract page ID with page title (namespace 0)
```bash
gunzip -c enwiki-20200220-page.sql.gz | sed s/"),("/"\n"/g | grep ",0," | sed -e s/"^INSERT.*("// | grep "^\([0-9]*\),0,.*" | sed s/"^\([0-9]*\),0,'\(.*\)','.*$"/"\1 \2"/ | sed s/"^\([0-9]*\) \(.*\)','.*$"/"\1 \2"/ > page.txt
```

Download [pagelinks sql dump](https://dumps.wikimedia.org/enwiki/20200220/enwiki-20200220-pagelinks.sql.gz), and extract pagelink from ID with pagelink to title (namespace 0)
```bash
gunzip -c enwiki-20200220-pagelinks.sql.gz | sed -e s/"),("/"\n"/g | grep ",0," | sed -e s/"^INSERT.*("// | sed -e s/");$"// | sed -e s/",0,"/" "/ | sed -e s/" '"/" "/ | sed -e s/"',.*$"/""/ > pagelinks.txt
```

Merge two files
```bash
awk 'NR==FNR { map[$2] = $1; next }{ if(map[$2]) print $1" "map[$2]; else print $1" "$2;}' page.txt pagelinks.txt > 20200220.tmp
```

Get rid of the lines that are not numeric
```bash
grep "^[0-9]* [0-9]*$" 20200220.tmp > 20200220.txt
```

**optional**: remove links which link to itself
```bash
awk '{if ($1 != $2)  print $1" "$2;}' 20200220.txt > 20200220.txt
```

Select some nodes ID and write it into nodes.txt with one node per line
```bash
shuf -i 2000-65000 -n 10 > nodes.txt
```
or select nodes from list
```python
node = ['1594759','848289','39654996']
f = open('nodes.txt','w')
for i in node:
    f.write(i+'\n')
f.close()
```

Select the row that 2 nodes both in nodes.txt
```bash
awk 'NR==FNR { a[int($0)]; next }{ if($1 in a) if($2 in a) print $1" "$2;}' nodes.txt 20200220.txt > links.txt
```
[reference1](https://libsq.tumblr.com/post/46678912694/parsing-wikipedias-pagelinks-sql-dump)
[reference2](https://www.it1352.com/313854.html)

----------------------------------

Extract title based on nodeID
```python
import json
import re
node = ['1594759','848289','39654996']

f = open("./enwiki-20200101-page.sql/enwiki-20200101-page.sql", encoding='utf-8')
lines = f.read()
f.close()

dic = {}
for i in node:
    result = re.search(r"\("+str(i)+",0,'(.*?)',",lines).group(1).replace('_',' ')
    dic[str(i)] = result
    print(i, result)

print(dic)
jsObj= json.dumps(dic)
fileObject = open('jsonFile.json', 'w')
fileObject.write(jsObj)
fileObject.close()
```

----------------------------------

Parsing Wikipedia xml dump:
```bash
awk '/<title>Leaf vegetable</,/<page>/' enwiki-20200201-pages-articles_1.xml > 1.txt
```
[reference](https://www.itranslater.com/qa/details/2120742941919544320)

-A -B -C 后面都跟阿拉伯数字，-A是显示匹配后和它后面的n行。-B是显示匹配行和它前面的n行。-C是匹配行和它前后各n行。
于是，
grep -A 4 wikipedia 密码文件.txt

就是搜索密码文件，找到匹配“wikipedia”字串的行，显示该行后后面紧跟的4行。

-n ：输出行号。
搜索test.log中满足123的内容的行号 grep -n '123' test.log
[reference](https://blog.51cto.com/3550334/787812)
[reference](https://blog.csdn.net/huashao0602/article/details/78018743)

sed -n "开始行，结束行p" 文件名，表示查看文件的开始行到结束行的内容，sed -n "5,9p" SpecialVariable.sh
[reference](https://blog.csdn.net/qq_29663071/article/details/79812252)