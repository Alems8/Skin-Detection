# Skin-Detection
I video usati come dataset per questa esperimento sono disponibili al link: https://feeval.org/Data-sets/Skin_Colors.html
Per realizzare l'esperimento sono stati usati diversi moduli per eseguire le varie azioni necessarie alla manipolazione del dataset originale per ottenerne uno utilizzabile
per far funzionare MLP.
Sono stati definiti 4 metodi per rendere il codice più leggibile ed evitare di avere duplicate alcune porzioni del codice.
Il programma si avvia indicando nel main nel metodo HaveAlreadyFrames se si hanno o meno dei frame con cui fare training e test. Dopodiché il resto del codice verrà eseguito 
automaticamente e riporterà l'accuracy score, la matrice di confusione e due immagini create dalla sovrapposizione dei risultati ottenuti dopo il training e i frame originali
in cui verranno colorate di nero tutte quelle zone che non contengono porzioni di pelle mentre saranno lasciate con l'immagine originale le altre.
Le due immagini ottenute saranno salvate in una cartella con il seguente indirizzo "D:\AI\ExamProject\Results" . E' quindi necessario cambiare il percorso di salvataggio che è 
presente nel metodo imageResults.

Metodi Presenti:
- HaveAlreadyFrames(answer, Framepath = None, MaskPath=None): in questo metodo si chiede se si hanno o meno dei frame con cui fare il training e i test. Di default si suppone 
che non si disponga di ciò quindi in caso contrario si deve indicare il percorso sia dei frame a colori che di quelli che fanno da maschera, contenedoli tutti in due liste 
separate.
Altrimenti si utilizzano dei video predefiniti, il cui percorso deve essere però modificato a meno che non sia identico a quello presente nel codice. Dopodiché attraverso il 
modulo moviepy si aprono i vari video e si salva un frame per ogni secondo del video. Per salvare i frame si utilizza il modulo os. Per far si che i frame salvati siano sempre 
gli stessi sia per i video a colori che per la maschera si confrontano le due durate e si sceglie quello con durata inferiore. Infine vengono ritornate due liste una contenente
i percorsi dei frame l'altro delle maschere.

-getArrays(titles): in questo metodo si creano due array contenenti le features e le labels che serviranno per effettuare l'addestramento successivamente. Si aprono i frame uno
e si ridimensionano ad una dimensione prefissata. Per aprire i frame si utilizza il modulo PIL e per convertire da RGB a HLS si utilizza un metodo già presente all'interno del 
modulo colorsys. I valori di HLS sono utilizzati per classificare i pixel. Per definire nella maschera, se un pixel sia bianco o nero si utilizza la L che se è inferiore a 0.1
viene considerato nero altrimenti bianco. Alla fine si ritornano due array del modulo numpy.

-compute(InputArray, LabelArray): In questo metodo tramite i due array contenenti features e labels si crea un dataframe del modulo pandas e da esso si estraggono x e y che poi 
attraverso un metodo di sklearn vengono divisi in dati per test e train con l'80% destinato all'addestramento e il restante 20% al test. Dopodiché si normalizzano i dati di 
input e si esegue il training. Si esegue il predict su alcuni pixel che poi saranno quelli utilizzati per creare le due immagini di output e si mostra la matrice di confusione 
attraverso il modulo matplotlib. Alla fine si ritornano i label predetti per le due immagini.

-def imageResults(array, i, y): Prende in ingresso la lista contenete i percorsi per accedere ai frame richiesti, i serve per salvare le immagini risultanti con nomi diversi. 
y corrisponde alle etichette predette in precedenza. 
