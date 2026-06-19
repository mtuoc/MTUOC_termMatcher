MTUOC_termMatcher

Example of use in a script:

```
from MTUOC_termMatcher import termMatcher
text="There are 3 main operating systems: Windows, Linux and Mac."
matcher = termMatcher(db_path="my_glossary.db", max_ngram=5)
matcher.load_tbx("MicrosoftTermCollection-catalan.tbx") #you only need to load once, then everything is stored in de Sqlite database.
resultats=matcher.search(text,"en-US",similarity_threshold=95)
for resultat in resultats:
    sourceterm=resultat["matched_db_term"]
    targetterm=resultat["translations"]["ca-ES"]
    print(sourceterm,targetterm)
```
