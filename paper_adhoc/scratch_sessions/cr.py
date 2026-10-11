import json,sys,urllib.request,urllib.parse,time
q=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 6
u="https://api.crossref.org/works?"+urllib.parse.urlencode({'query.bibliographic':q,'rows':n,'select':'DOI,title,author,container-title,published,volume,article-number,page,issue'})
for k in range(5):
    try:
        d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'litcheck/1.0'}))); break
    except Exception as e:
        time.sleep(4)
else:
    print('FAILED',q); sys.exit(0)
print('##',q)
for it in d['message']['items']:
    print(it.get('DOI'),'|',(it.get('title') or [''])[0],'|',', '.join(a.get('family','') for a in it.get('author',[])),'|',(it.get('container-title') or [''])[0],it.get('volume'),it.get('issue'),it.get('article-number') or it.get('page'),it.get('published',{}).get('date-parts'))
time.sleep(2)
