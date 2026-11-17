# Zadanie 2 – klasyfikacja tekstu (sprawozdanie)

Nazwisko na M, więc z listy sklearn wziąłem Logistic Regression i MLP.
Zbiór zewnętrzny: AG News (4 klasy). Do lokalnych TODO dałem klasy: praca / dom / nauka.

Przed klasyfikacją tekst idzie przez `clean_text` z zadania 1 (w komendzie jest obowiązkowe `cleaning`).

## Co jest w bocie

- `/prepare_data_for_classification agnews` albo `todo`
- `/classifier cleaning "tekst" bow|tfidf|embedding logreg|mlp`
- `/evaluate tfidf mlp` albo `/evaluate all`
- `/validate` – moje zdania spoza zbioru
- `/label <id> <klasa>`

## 4. Reprezentacje – parametry

### BoW (CountVectorizer)

- lowercase=True
- ngram_range=(1, 2) – zwykłe słowa + bigramy
- max_features=5000 – żeby słownik nie urósł za bardzo
- min_df=2 – wyrzucam hapaxy

BoW nie ogarnia synonimów, tylko zlicza tokeny. Macierz jest sparse. Gdyby brać np. QDA / Gaussian Process, trzeba by `toarray()` albo TruncatedSVD. U mnie logreg i mlp radzą sobie ze sparse, więc tego nie robiłem.

### TF-IDF (TfidfVectorizer)

Te same ngramy / max_features / min_df co w BoW, plus:
- use_idf=True
- smooth_idf=True
- sublinear_tf=False

Ogólnie podobne ograniczenia jak BoW, ale częste słowa dostają mniejszą wagę, co na newsach ma sens.

### Embedding (sentence-transformers, all-MiniLM-L6-v2)

Dostaję gęsty wektor zdania (384 dim) i podaję go do klasyfikatora.

Uwaga z zajęć: MultinomialNB odpada, bo embeddingi mogą mieć wartości ujemne. Dlatego i tak zostałem przy logreg/mlp (pasują też do litery M). Model jest pod angielski, więc do AG News pasuje, do polskich TODO już gorzej.

## 5. Podział i walidacja poza zbiorem

Podział:
```
train_test_split(..., test_size=0.2, stratify=labels, random_state=42)
```
80/20, stratyfikacja, stałe ziarno. Jak któraś klasa ma mniej niż 2 przykłady (małe TODO), to stratify wyłączam, bo sklearn wtedy wywala błąd.

Wymyśliłem 4 zdania w stylu AG News (nie brałem ich z train):

1. Olympic athletes win gold medals in swimming finals -> Sports
2. Central bank raises interest rates amid inflation fears -> Business
3. Astronomers discover a new exoplanet near a red dwarf -> Sci/Tech
4. Diplomats meet to discuss ceasefire in the region -> World

Wyniki (`/validate`):

embedding + logreg: 4/4
- Sports ok (p~0.91)
- Business ok (p~0.96)
- Sci/Tech ok (p~0.81)
- World ok (p~1.00)

tfidf + logreg: 3/4
- Sci/Tech poszło jako World (p~0.40). Krótkie zdanie, mało „technicznych” słów z korpusu, TF-IDF się pogubił.

Komentarz: embedding lepiej trzyma się sensu przy nowych zdaniach. TF-IDF bardziej zależy od tego, czy konkretne słowa już widział.

## 6. Metryki

Dla wielu klas liczę one-vs-rest (TP/TN/FP/FN per klasa) i potem macro / weighted.
Wzory jak w instrukcji: ACC, ERR, Precision, Recall, F1.
W bocie: `/evaluate ...` albo `/evaluate all`.

## 7. Dane

AG News (`sh0416/ag_news`). Na klasę biorę 2000 przykładów (razem 8000), random_state=42.
Tekst = title + description.

## 9. Porównanie 6 konfiguracji

| repr | method | ACC | ERR | F1 macro |
|---|---|---:|---:|---:|
| bow | logreg | 0.860 | 0.140 | 0.860 |
| bow | mlp | 0.858 | 0.142 | 0.857 |
| tfidf | logreg | 0.886 | 0.114 | 0.886 |
| tfidf | mlp | 0.854 | 0.146 | 0.854 |
| embedding | logreg | 0.889 | 0.111 | 0.889 |
| embedding | mlp | 0.870 | 0.130 | 0.870 |

### Macierze

bow + logreg
```
true\pred         Business    Sci/Tech      Sports       World
Business               331          45           6          18
Sci/Tech                45         330          12          13
Sports                  11           4         371          14
World                   17          17          22         344
ACC = 0.860 | ERR = 0.140 | F1 macro = 0.860
```

bow + mlp
```
true\pred         Business    Sci/Tech      Sports       World
Business               325          45           7          23
Sci/Tech                47         326          14          13
Sports                   9           5         373          13
World                   17          16          19         348
ACC = 0.858 | ERR = 0.142 | F1 macro = 0.857
```

tfidf + logreg
```
true\pred         Business    Sci/Tech      Sports       World
Business               338          39          10          13
Sci/Tech                26         347          13          14
Sports                   9           5         377           9
World                   16          10          18         356
ACC = 0.886 | ERR = 0.114 | F1 macro = 0.886
```

tfidf + mlp
```
true\pred         Business    Sci/Tech      Sports       World
Business               323          48           6          23
Sci/Tech                47         326          12          15
Sports                   8           5         374          13
World                   22          16          18         344
ACC = 0.854 | ERR = 0.146 | F1 macro = 0.854
```

embedding + logreg
```
true\pred         Business    Sci/Tech      Sports       World
Business               341          41           0          18
Sci/Tech                40         342           6          12
Sports                   5           4         384           7
World                   20          12          13         355
ACC = 0.889 | ERR = 0.111 | F1 macro = 0.889
```

embedding + mlp
```
true\pred         Business    Sci/Tech      Sports       World
Business               330          48           5          17
Sci/Tech                44         333           6          17
Sports                   4           2         380          14
World                   21          15          15         349
ACC = 0.870 | ERR = 0.130 | F1 macro = 0.870
```

### Co wyszło najlepiej

Najlepiej: **embedding + logreg** (ACC 0.889, F1 0.889).

Embedding lepiej łapie temat newsa niż samo zliczanie słów. Logreg na tych wektorach był stabilniejszy niż mlp. TF-IDF + logreg jest bardzo blisko (0.886) i szybszy, więc też okej. MLP na BoW/TF-IDF nie dało zysku. Na zdaniach spoza zbioru embedding też wygrał (4/4 vs 3/4).
