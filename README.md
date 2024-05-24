# RakutenPjct
![image](https://github.com/6racuse/RakutenPjct/assets/148326846/7026c66e-bdc6-48ca-8c67-5fdd0d7ac434)

Ce projet a pour étude le dataset proposé dans le cadre du **Rakuten France Multimodal Product Data Classification Project**, proposé par Rakuten Institute of Technology, Paris et lancé le 6 janvier 2020. Ce défi se concentre sur la classification multimodale (texte et image) à grande échelle des codes types de produits, avec pour objectif de prédire le code type de chaque produit tel que défini dans le catalogue de Rakuten France.

Nous, Welto et Alexis, proposons ici une solution viable au regard de nos connaissances et capacités en Data Science, suivant les cours suivis à Centrale Casablanca, dans le cadre de notre semestre universitaire à l'étranger.

- [Résultats](#Résultats)
- [Setup](#Setup)
- [Run](#Run)
- [History](#History)

Nous avons séparé le projet en plusieurs parties **distingues** mais pour autant **complémentaires** : 
- Préprocessing, Feature engineering, Affichage des données et compréhension du jeu de données.
- Elaboration d'un éventail des méthodes potentiellement viables pour répondre au problème
- Proposition d'un modèle couplé complexe permettant d'obtenir la meilleure réponse au problème

## Résultats
La dernière version du projet (e1r0) propose un **f1_score** de **0.8256**, obtenu avec la méthode SVM 
## Setup

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances

Clonez le dépôt et installez les dépendances :

```bash
git clone https://github.com/6racuse/RakutenPjct.git
cd RakutenPjct
pip install -r requirements.txt
```
Même si le code gère l'instalation des ressources nltk, nous recommandons d'exécuter les instructions suivantes :

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```
## Run 



## History
Ici est récaptilé l'avancé du projet pour chacun des membres du groupe
### Alexis CHATAIL--RIGOLLEAU
- **15/05** : lancement du projet, préprocessing de la donnée textuelle "designation", récupération de la matrice TFIDF.
- **16,17/05** : Création du Notebook python pour tester les méthodes, définition de la stratégie en groupe, travail en local. Test de la méthode KNN, best param à K=49, f1_score = 0.68.
- **19/05** : Création du fichier .py main du projet, stockage de la matrice TFIDF en mémoire pour éviter de le reload à chaque fois. Méthode Random Forest longue en CV, et f1_score = 0.697.
- **20/05** : Résolution du problème de transfert des modèles (tout seul), en coupant le fichier .joblib en n fichiers binaires. Ajout d'une première méthode de Deep Learning sur une partition du fichier de départ, f1_score = 0.732.
- **21/05** : Travail en local sur une approche SVM. Début de la modélisation complexe de la solution, alliant plusieures méthodes. Amélioration de la solution Deep Learning. Amélioration de la structure du code de rendu en local. f1_score = 0.7917
- **22/05** : problème de commit trop gros : travail en local.
- **23/05** : problème réglé, retour au fichier main.py et amélioration de l'interface
- **24/05** : Amélioration de la récupération des données designation, et récupération des données description pour les utiliser. Mise en forme du réseau de neurone sous forme pipeline.

### Welto CANDE
- **15/05** : Démarrage du projet, importation et premier preprocessing des données textuelles (récupération des données d'intérêt). Travail en local.
- **18/05** : Travail en local, calcul des matrices TF-IDF et preprocessing (tokenisation, cleaning) , premiers pas vers l'application de SVM.
- **21/05** : Notebook de test : SVM, premier entraînement du modèle pour obtenir des premiers résultats de score. Score très bas : la prédiction des labels (X_test depuis le site) se basait sur un Y_test calculé depuis Y_train. 
- **22/05** : Création d'un fichier .py général. Entraînement du SVM avec des données train_test_split(), score plus acceptable : f1_score = 0.786
- **23/05** : Variation des hyperparamètres, effet du paramaètre "random_state" de train_test_split (recherche du f1_score max). Pour random_state = 53, f1_score = 0.800. Refonte du code général pour obtenir la prédiction des labels de X_test (.csv du site du challenge). Introduction d'une cross validation. Après calcul des paramètres optimaux, f1_score = 0.8197
- **24/05** : Implémentation d'un nouvel outil de tokenisation (nltk) bien plus rapide. Travail sur une approche Gradient Boosting. À ce stade, f1_score = 0.8256
