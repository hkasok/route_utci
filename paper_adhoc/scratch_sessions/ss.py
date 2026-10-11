import json,sys,urllib.request,time
for doi in sys.argv[1:]:
    try:
        d=json.load(urllib.request.urlopen("https://api.semanticscholar.org/graph/v1/paper/DOI:"+doi+"?fields=title,year,venue,abstract,authors,tldr"))
        print('==',doi,'|',d.get('title'),'|',d.get('year'),'|',d.get('venue'),'|',', '.join(a['name'] for a in d.get('authors',[])))
        print(d.get('abstract') or ('TLDR: '+str((d.get('tldr') or {}).get('text'))))
    except Exception as e: print('==',doi,'ERR',e)
    time.sleep(1.5)
