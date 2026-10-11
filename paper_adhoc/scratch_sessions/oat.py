import sys, json, urllib.request, urllib.parse
for q in sys.argv[1:]:
    url = "https://api.openalex.org/works?per_page=2&search="+urllib.parse.quote(q)
    try: res = json.load(urllib.request.urlopen(url))["results"]
    except Exception as e: print(q,"ERR",e); continue
    for r in res[:1]:
        inv = r.get("abstract_inverted_index") or {}
        words = sorted([(p,w) for w,ps in inv.items() for p in ps])
        au = ", ".join(a["author"]["display_name"] for a in r["authorships"][:6])
        src = (r.get("primary_location") or {}).get("source") or {}
        b = r.get("biblio",{})
        print("=== ", r.get("doi"), "|", r["title"], "|", au, "|", r["publication_year"], "|", src.get("display_name"), b.get("volume"), b.get("first_page"), "-", b.get("last_page"))
        print(" ".join(w for _,w in words)[:2500]); print()
