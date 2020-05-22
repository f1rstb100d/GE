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

Select the row that 2 nodes both in nodes.txt
```bash
awk 'NR==FNR { a[$0]; next }{ if($1 in a) if($2 in a) print $1" "$2;}' nodes.txt 20200220.txt > links.txt
```
[reference1](https://libsq.tumblr.com/post/46678912694/parsing-wikipedias-pagelinks-sql-dump)

[reference2](https://www.it1352.com/313854.html)
