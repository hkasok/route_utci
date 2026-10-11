import json,sys,urllib.request
for doi in sys.argv[1:]:
    try:
        d=json.load(urllib.request.urlopen("https://api.openalex.org/works/doi:"+doi))
        inv=d.get('abstract_inverted_index') or {}
        words=sorted(((p,w) for w,ps in inv.items() for p in ps))
        print('==',doi,'|',d.get('title'),'|',d.get('publication_year'))
        print(' '.join(w for p,w in words) or 'NO ABSTRACT')
    except Exception as e: print('==',doi,'ERR',e)
