# RakutenPjct
![image](https://github.com/6racuse/RakutenPjct/assets/148326846/7026c66e-bdc6-48ca-8c67-5fdd0d7ac434)

Ce projet a pour étude le dataset proposé dans le cadre du **Rakuten France Multimodal Product Data Classification Project**, proposé par Rakuten Institute of Technology, Paris et lancé le 6 janvier 2020. Ce défi se concentre sur la classification multimodale (texte et image) à grande échelle des codes types de produits, avec pour objectif de prédire le code type de chaque produit tel que défini dans le catalogue de Rakuten France.

Nous, Welto et Alexis, proposons ici une solution viable au regard de nos connaissances et capacités en Data Science, suivant les cours suivis à Centrale Casablanca, dans le cadre de notre semestre universitaire à l'étranger.

- [Résultats](#Résultats)
- [Setup](#Setup)
- [Run](#Run)
- [History](#History)
- [Exemples](#Exemples)
- [Contacts](#Contacts)

Nous avons séparé le projet en plusieurs parties **distingues** mais pour autant **complémentaires** : 
- Préprocessing, Feature engineering, Affichage des données et compréhension du jeu de données.
- Elaboration d'un éventail des méthodes potentiellement viables pour répondre au problème
- Proposition d'un modèle couplé complexe permettant d'obtenir la meilleure réponse au problème

## Résultats
La dernière version du projet (**e2r0**) propose un **f1_score** de **0.8256**, obtenu avec la méthode SVM, avec les hyperparamètres `C=8.071428571428571, gamma=0.1, kerrnel:'rbf' `
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
Même si le code gère l'installation des ressources nltk, nous recommandons d'exécuter les instructions suivantes :

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Récupération des modèles

**Le repository ne comprend pas les modèles pré-entrainnés** 

Certains modèles sont un peu longs à entrainner. Si vous n'avez pas le temps, nous avons créé un lien Wetransfer avec tous les modèles pré-entrainnés (disponible du 27/05/2024 au 03/05/2024, nous [contacter](#Contacts) si besoin pour les réactualiser) :

`lien hypertexte`

Il est évident que le code fonctionne très bien sans, il prendra juste plus de temps à donner renvoyer les labels.
## Run 

Pour lancer le programme, exécuter le programme **Zmain.py** dans le path dans lequel le repository a été copié :
```PS
PS C:\Codes\IA\DS\Rak\RakutenPjct>
```

Suite à cela, le programme effectue le preprocessing des données, et l'onglet de sélection du modèle s'affiche : 

```
 _______          __               _                          _      _____    _______                   _               _    
|_   __ \        [  |  _          / |_                       / \    |_   _|  |_   __ \                 (_)             / |_  
  | |__) |  ,--.  | | / ] __   _ `| |-'.---.  _ .--.        / _ \     | |      | |__) | .--.  .--.     __ .---.  .---.`| |-' 
  |  __ /  `'_\ : | '' < [  | | | | | / /__\\[ `.-. |      / ___ \    | |      |  ___[ `/'`\] .'`\ \  [  / /__\\/ /'`\]| |   
 _| |  \ \_// | |,| |`\ \ | \_/ |,| |,| \__., | | | |    _/ /   \ \_ _| |_    _| |_   | |   | \__. |_  | | \__.,| \__. | |,  
|____| |___\'-;__[__|  \_]'.__.'_/\__/ '.__.'[___||__]  |____| |____|_____|  |_____| [___]   '.__.'[ \_| |'.__.''.___.'\__/  
                                                                                                        \____/

Choisir un modèle à éxécuter : 

     1 - Neural Network (f1-score 0.808) 
     2 - SVM (f1-score 0.8256) 
     3 - KNN (f1-score 0.69)
     4 - Solution to the project

Choix :   
```
Il suffit alors de choisir le modèle, suite à quoi une nouvelle ligne apparaît, proposant de load s'il existe le modèle pré-entrainné (voir les [exemples](#Exemples))

## History
Ici est récapitulé l'avancé du projet pour chacun des membres du groupe
### Alexis CHATAIL--RIGOLLEAU
- **15/05** : lancement du projet, préprocessing de la donnée textuelle "designation", récupération de la matrice TFIDF. travail sur la branch **6racuse's-work** 
- **16,17/05** : Création du Notebook python pour tester les méthodes, définition de la stratégie en groupe, travail en local. Test de la méthode KNN, best param à K=49, f1_score = 0.68.
- **19/05** : Création du fichier .py main du projet, stockage de la matrice TFIDF en mémoire pour éviter de le reload à chaque fois. Méthode Random Forest longue en CV, et f1_score = 0.697.
- **20/05** : Résolution du problème de transfert des modèles (tout seul), en coupant le fichier .joblib en n fichiers binaires. Ajout d'une première méthode de Deep Learning sur une partition du fichier de départ, f1_score = 0.732. *version e0r0*
- **21/05** : Travail en local sur une approche SVM. Début de la modélisation complexe de la solution, alliant plusieures méthodes. Amélioration de la solution Deep Learning. Amélioration de la structure du code de rendu en local. f1_score = 0.7917. *version e1r0*
- **22/05** : problème de commit trop gros : travail en local.
- **23/05** : problème réglé, retour au fichier main.py et amélioration de l'interface utilisateur, test d'ajout du pipeline neural network.
- **24/05** : Amélioration de la récupération des données designation, et récupération des données description pour les utiliser. Mise en forme du réseau de neurone sous forme pipeline. 
- **25/05** : Multiples gestion de merges entre les branches. Tests de régression non concluants => récupération du commit *e1r0* et épluchage des commits successifs. Fermeture de la branch **6racuse's-work**, push de la *version* release *e1r1* passant (enfin) le test de régression sur le **main**, suppression de l'enregistrement des matrices data, *version e1r2* . Merge de code de la branch **welto** utilisant nltk pour accélérer le préprocessing des strings, *version e1r3*

### Welto CANDE
- **15/05** : Démarrage du projet, importation et premier preprocessing des données textuelles (récupération des données d'intérêt). Travail en local.
- **18/05** : Travail en local, calcul des matrices TF-IDF et preprocessing (tokenisation, cleaning) , premiers pas vers l'application de SVM.
- **21/05** : Notebook de test : SVM, premier entraînement du modèle pour obtenir des premiers résultats de score. Score très bas : la prédiction des labels (X_test depuis le site) se basait sur un Y_test calculé depuis Y_train. 
- **22/05** : Création d'un fichier .py général. Entraînement du SVM avec des données train_test_split(), score plus acceptable : f1_score = 0.786
- **23/05** : Variation des hyperparamètres, effet du paramaètre "random_state" de train_test_split (recherche du f1_score max). Pour random_state = 53, f1_score = 0.800. Refonte du code général pour obtenir la prédiction des labels de X_test (.csv du site du challenge). Introduction d'une cross validation. Après calcul des paramètres optimaux, f1_score = 0.8197
- **24/05** : Implémentation d'un nouvel outil de tokenisation (nltk) bien plus rapide. Travail sur une approche Gradient Boosting. À ce stade, f1_score = 0.8256
- **25/05** : Ajout d'un pipeline de preprocessing pour la fluidification des process et la simplification du code : Tokenisation avec nltk, vectorisation et entraînement du modèle SVM. Test de la solution XGB (eXtreme Gradient Boosting), trop gourmand en ressources et temps de calcul.
- **26/05** : Travail sur le code final, merge du code kNN sur le code Zmain.py. Travail sur le rapport final : construction de courbes d'erreur et de score pour le modèle SVM.

  
## Exemples

### Use Case 1 : Neural Network model 
Après avoir lancé le programme, sur l' [onglet de sélection](#Run), rentrer :
```PS
Choix : 1
```
Puis la fenêtre suivante apparaît :
```
Reload neural network model - mandatory if nn_model.keras doesn't exist - (yes/no) ? : 
```
- **yes** recrée l'entrainnement du réseau de neurones.
- **no**  charge le model entrainné s'il est présent en mémoire, sinon il relance l'entrainnement du réseau de neurones

Une fois le choix fait, le programme effectue la prédiction des labels, et les stocke dans le fichier **output_nn.csv** du répertoire **output**. Le modèle entrainné peut être récupéré sous le nom de **nn_model.keras** dans le répertoire **models**


## Contacts
