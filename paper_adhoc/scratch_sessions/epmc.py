import json,sys,urllib.request,urllib.parse
for doi in sys.argv[1:]:
    u="https://www.ebi.ac.uk/europepmc/webservices/rest/search?"+urllib.parse.urlencode({'query':'DOI:"%s"'%doi,'resultType':'core','format':'json'})
    try:
        d=json.load(urllib.request.urlopen(u))
        r=d['resultList']['result']
        print('==',doi, '|', (r[0].get('title') if r else 'NOT FOUND'))
        if r: print(r[0].get('abstractText','NO ABSTRACT'))
    except Exception as e: print('==',doi,'ERR',e)
