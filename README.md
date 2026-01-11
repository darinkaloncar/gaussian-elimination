# Gaussian Elimination – HPC Project

## Opis problema

Ovaj projekat se bavi rešavanjem sistema linearnih jednačina oblika:

A · x = b

gde je A kvadratna matrica dimenzije N × N, b vektor dimenzije N, a x vektor nepoznatih.  
Problem se rešava primenom **Gausovog metoda eliminacije sa parcijalnim pivotiranjem**.

Cilj projekta je:
- implementacija sekvencijalnog i paralelnog rešenja u Python-u i Rust-u,
- analiza performansi paralelizacije kroz eksperimente jakog i slabog skaliranja,
- poređenje dobijenih ubrzanja sa teorijskim granicama (Amdahlov i Gustafsonov zakon),
- vizualizacija toka rešavanja problema po iteracijama.

---

## Opis algoritma

Gausov metod eliminacije transformiše polazni sistem linearnih jednačina u ekvivalentan sistem čija je matrica gornje trougaona. Proces se sastoji iz sledećih koraka:

1. **Parcijalno pivotiranje**  
   U svakoj iteraciji bira se pivot element maksimalne apsolutne vrednosti u tekućoj koloni i vrši se zamena redova radi numeričke stabilnosti.

2. **Eliminacija ispod pivota**  
   Elementi ispod pivot elementa se poništavaju oduzimanjem odgovarajućih linearnih kombinacija redova.

3. **Iterativno ponavljanje**  
   Postupak se ponavlja za svaku kolonu matrice dok se ne dobije gornje trougaona matrica, tj. kvadratna matrica čiji su svi elementi ispod glavne dijagonale jednaki nuli.

4. **Unazadna supstitucija**  
   Nakon eliminacije, rešenje sistema se dobija primenom unazadne supstitucije.

Proces eliminacije se odvija iterativno, pri čemu svaka iteracija predstavlja jedno stanje sistema koje se može sačuvati u datoteku radi kasnije analize i vizualizacije.

---

## Paralelizacija problema

Iako Gausov metod eliminacije sadrži sekvencijalni deo (izbor pivota i zamena redova), najveći deo računanja odnosi se na eliminaciju redova ispod pivota, koja se može efikasno paralelizovati.

---

## Eksperimenti skaliranja

U okviru projekta sprovode se sledeći eksperimenti:

- **Jako skaliranje (Strong scaling)**  
  Veličina problema ostaje konstantna, dok se menja broj procesorskih jezgara.

- **Slabo skaliranje (Weak scaling)**  
  Veličina problema se povećava proporcionalno broju procesorskih jezgara kako bi se zadržao približno konstantan posao po jezgru.

Dobijeni rezultati se porede sa teorijskim ubrzanjima definisanim:
- Amdahlovim zakonom (jako skaliranje)
- Gustafsonovim zakonom (slabo skaliranje)

---

## Vizualizacija

Na osnovu podataka generisanih tokom iteracija Gausovog metoda eliminacije, vrši se vizualizacija:
- promena stanja matrice tokom eliminacije,
- performansi paralelizacije u vidu grafika ubrzanja.


