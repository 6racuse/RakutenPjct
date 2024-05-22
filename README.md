# RakutenPjct


Ce projet a pour étude le dataset proposé dans le cadre du **Rakuten France Multimodal Product Data Classification Project**, proposé par Rakuten Institute of Technology, Paris et lancé le 6 janvier 2020. Ce défi se concentre sur la classification multimodale (texte et image) à grande échelle des codes types de produits, avec pour objectif de prédire le code type de chaque produit tel que défini dans le catalogue de Rakuten France.

Nous, Welto et Alexis proposons ici une solution viable au regard de nos connaissances et capacités en Data Science, suivant les cours suivis à Centrale Casablanca, dans le cadre de notre semestre universitaire à l'étranger.

Nous avons séparé le projet en plusieurs parties **distingues** mais pour autant **complémentaires** : 
- Préprocessing, Feature engineering, Affichage des données et compréhension du jeu de données.
- Elaboration d'un éventail des méthodes potentiellement viables pour répondre au problème
- Proposition d'un modèle couplé complexe permettant d'obtenir la meilleure réponse au problème

## Project History
### Alexis CHATAIL--RIGOLLEAU
- 15/05 : lancement du projet, préprocessing de la donnée textuelle "designation", récupération de la matrice TFIDF.
- 16,17/05 : Création du Notebook python pour tester les méthodes, définition de la stratégie en groupe, travail en local.
- 19/05 : Création du fichier .py main du projet, stockage de la matrice TFIDF en mémoire pour éviter de le reload à chaque fois
- 20/05 : Résolution du problème de transfert des modèles (tout seul), en coupant le fichier .joblib en n fichiers binaires 
- 21/05 : Travail en local sur une approche SVM. Début de la modélisation complexe de la solution, alliant plusieures méthodes. Amélioration de la solution Deep Learning. Amélioration de la structure du code de rendu en local.


  ![image](https://github.com/6racuse/RakutenPjct/assets/148326846/07db6a81-180b-4600-b3d4-c2d9ea86932b)
