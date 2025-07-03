# Package Kedro : `mnist_backend_kedro`

Ce répertoire contient le code source de notre pipeline d'entraînement de modèle, structuré avec le framework [Kedro](https://kedro.readthedocs.io/).

## Rôle dans l'architecture MLOps

Ce projet Kedro est le cœur de notre **Continuous Training (CT)**. Il représente la **maturité MLOps de niveau 1**, où l'entraînement du modèle est automatisé et reproductible.

- **Pipeline d'entraînement** : Il définit de manière explicite toutes les étapes nécessaires pour passer des données brutes (`raw data`) à un modèle entraîné :
    - `pipelines/data_preparation`: Nœuds et pipeline pour le nettoyage, la transformation et la division des données MNIST.
    - `pipelines/training`: Nœuds et pipeline pour l'entraînement du modèle `ConvNet`, sa validation et sa sérialisation.
- **Reproductibilité** : En versionnant le code de ce pipeline (via Git) et les données (via DVC ou autre), nous nous assurons que n'importe quel entraînement de modèle peut être reproduit à l'identique.
- **Automatisation** : Ce pipeline est conçu pour être exécuté automatiquement. Dans notre architecture MLOps niveau 2, le pipeline CI/CD (défini dans `mnist-deployment`) peut déclencher une exécution de ce pipeline Kedro (par exemple, chaque semaine ou lorsque de nouvelles données sont disponibles) pour produire un nouveau candidat de modèle.

## Structure

- `pipelines/`: Contient les différentes parties de notre pipeline (préparation des données, entraînement).
- `pipeline_registry.py`: Enregistre et assemble les pipelines modulaires.
- `model/`: Contient la définition de notre modèle PyTorch (`convnet.py`).
- `settings.py`: Fichier de configuration principal du projet Kedro.

Pour exécuter le pipeline complet :

```bash
cd kedro/
kedro run
``` 