





data.py
- Fonction pour copier/coller des images depuis raw data vers 100_data
  > Depuis le terminal, taper "make run_100_data"
  > Copie/colle 200 images du training_set dont 100 positives et 100 négatives à la DMLA
  > Copie/colle 60 images du validation_set dont 30 positives et 300 négatives à la DMLA
  > Copie/colle 20 images du test_set dont 100 positives et 20 négatives à la DMLA





*************************************************************
TUTORIELS
PACKAGER NIVEAU 1

Dans un notebook
Créer la fonction

Dans un fichier .py
Copier / coller la fonction sans argument (à voir plus tard avec le main)
Attention aux docstrings qui doivent être bien indentées.
vérifier les imports en haut du fichier .py et les reporter dans requirements

Dans le Makefile
Créer dans la bonne section une ligne similaire :
"""run_100_images:
	  python -c 'from dmla.ml_logic.data import copier_coller; copier_coller()' """

Dans le terminal
make run_100_images

TUTORIEL GIT
git checkout branche_locale >> se rendre dans sa branche locale

git status >> voir l'état des lieux entre la branche locale et sa remote (même branche dans github)

git add nom_du_fichier >> pour sauvegarder une nouvelle version de git local vers git push

git commit -m "message" >> mise à jour de la branche locale

git status >> vérifier les mises à jour

git push origin branche_locale

git pull

git branch >> donne l'indication sur la branche
