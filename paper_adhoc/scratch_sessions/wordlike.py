"""Word-processor style count of the abstract: units are separate words."""
import re, sys
s=open(sys.argv[1]).read()
s=re.sub(r"\\SIrange\{([^}]*)\}\{([^}]*)\}\{[^}]*\}", r"\1–\2 U", s)
s=re.sub(r"\\SI\{([^}]*)\}\{[^}]*\}", r"\1 U", s)
s=re.sub(r"\\num\{([^}]*)\}", r"\1", s)
s=re.sub(r"\\[a-zA-Z]+", "", s)
print(len(s.split()))
