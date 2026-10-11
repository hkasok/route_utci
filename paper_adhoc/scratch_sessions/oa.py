import sys, json, urllib.request
for doi in sys.argv[1:]:
    try:
        r = json.load(urllib.request.urlopen("https://api.openalex.org/works/https://doi.org/"+doi))
    except Exception as e:
        print(doi, "ERR", e); continue
    inv = r.get("abstract_inverted_index") or {}
    words = sorted([(p,w) for w,ps in inv.items() for p in ps])
    au = ", ".join(a["author"]["display_name"] for a in r["authorships"][:6])
    src = (r.get("primary_location") or {}).get("source") or {}
    b = r.get("biblio",{})
    print("=== ", doi, "|", r["title"], "|", au, "|", r["publication_year"], "|", src.get("display_name"), b.get("volume"), b.get("first_page"), "-", b.get("last_page"))
    print(" ".join(w for _,w in words)[:6000]); print()
