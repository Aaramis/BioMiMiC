# BioMiMiC

Pour le moment c'est pas du tout propre.
Y a un notebook pour faire 2-3 modifs rapides des datasets que tu m'as envoyé.

Puis y a 2 fichiers pythons :
- 1 pour finetuner SMILES-BERT
- 1 pour predire un fichier csv en entrée

Pour le moment les fichiers sont pas du tout propre, pas optimisé et pas maxi-robuste face à des modifications (genre beaucoup de variables codées en dur).
Mais je reviendrai dessus une fois le test de screening finit et le dossier du projet complété.

## I. Introduction

## II. Getting Start

To start working on this project, follow these steps:

### Create environment

* Clone the repository: ```git clone git@github.com:Aaramis/BioMiMiC.git```
* Create a virtual environment: ```conda env create --file environment.yml```
* Activate the virtual environment: ``` conda activate hugging_face ``` 

### Update environment

* Using conda: ```conda env export > environment.yml```

## III. Optional Arguments

### [Predict File :](./SMILES-BERT/predict.py)

| Argument          | Description                      |
|-------------------|----------------------------------|
| --COCONUT         | "Flag to trigger COCONUT         |