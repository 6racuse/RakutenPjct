# RakutenPjct

Ce project traite de la classification de produit, en traitant des données textuelles. Le but est de prédire chaque code produit comme défini dans le catalogue Rakuten France, et est dérvié d'un Data Challenge proposé par Rakuten Institute of Technology, Paris.

## Project History
### Alexis CHATAIL--RIGOLLEAU
- 15/05 : lancement du projet, préprocessing de la donnée textuelle "designation", récupération de la matrice TFIDF.
- 16,17/05 : Création du Notebook python pour tester les méthodes, définition de la stratégie en groupe, travail en local.
- 19/05 : Création du fichier .py main du projet, stockage de la matrice TFIDF en mémoire pour éviter de le reload à chaque fois
- 20/05 : Résolution du problème de transfert des modèles (tout seul), en coupant le fichier .joblib en n fichiers binaires 

On se dit que le vocabulaire de la designation permettra de distinguer un produit d'un autre. 
![image](https://github.com/6racuse/RakutenPjct/assets/148326846/9b731f87-aa04-4e8c-ae90-59d0c1a0fd63)
