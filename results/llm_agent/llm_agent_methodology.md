# GoFlow — Intégration d'un LLM instruct local

XGBoost fournit les prévisions d'occupation à +15 minutes. Le LLM ne fait pas de
forecasting : il reçoit les cinq états déjà sauvegardés par le notebook 08, avec
mesures, tendances passées et voisins, puis propose une catégorie de risque, une
action et une justification. Les règles constituent le fallback. Aucun scénario
n'est recréé, aucune cible future n'est passée au modèle, aucun modèle trafic
n'est réentraîné. Les fichiers de results/decision_agent/ sont lus, pas modifiés.

## Modèle et environnement

Modèle : Qwen/Qwen2.5-0.5B-Instruct, environ 0,49 milliard de paramètres, issu du
[dépôt du fournisseur](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
La révision téléchargée et les SHA256 de chaque fichier sont enregistrés sous
models/qwen2.5-0.5b-instruct/download_provenance.json. Les informations effectives
de chargement, nombre exact de paramètres, mémoire RSS et versions sont enregistrées
dans llm_model_info.json. Ce modèle est préentraîné/instruct, sans adaptation à PEMS08.

L'environnement .venv-llm utilise --system-site-packages pour lire les dépendances
existantes sans les modifier. Aucune installation de package n'est nécessaire.
Le CPU est utilisé, avec quatre threads ; CUDA n'est pas disponible sur cette
installation PyTorch. Le backend retenu est le binaire CPU Windows officiel
[llama.cpp b11596](https://github.com/ggml-org/llama.cpp/releases/tag/b11596), avec le
[GGUF Qwen officiel Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).
Provenance et SHA256 du GGUF et du binaire sont enregistrés sous
models/qwen2.5-0.5b-instruct-gguf/download_provenance.json. Le serveur est lié à
127.0.0.1 sur un port libre et arrêté en fin de notebook. Sa RSS est mesurée
séparément de celle du noyau Python. Le chargement exige 1 Go libre.

Le premier essai Transformers/INT8 a généré cinq réponses, avec zéro JSON valide,
cinq fallbacks et trois délais dépassés. Tous ses résultats, textes, erreurs et
temps sont conservés dans diagnostics/transformers_int8_first_attempt. Le changement
de backend répond à cet échec technique, sans consulter les cibles futures et
sans modifier les scénarios. Aucun échec initial n'est effacé.

## Génération et validation

Chat template embarqué dans le GGUF officiel, prompt système français, température
zéro, maximum 300 nouveaux tokens et génération contrainte par schéma JSON dans
llama.cpp. Le schéma impose les champs et enums, pas les décisions ni une raison
préécrite. Le texte utile est récupéré dans message.content.
Le prompt et les paramètres sont fixés avant de parcourir les cinq scénarios ;
aucun réglage par comparaison à la cible future n'est réalisé. Le modèle n'est
pas entraîné pendant cette étape. Un timeout HTTP de 60 secondes et un contrôle
de durée après réponse déclenchent un fallback ; une opération serveur en cours
n'est pas garantie d'être interrompue instantanément. Le log serveur est conservé.
Le texte brut, le JSON extrait, les erreurs, motifs de fallback,
tokens et durées restent disponibles pour chacun des cinq appels réels.

L'extraction accepte du bruit autour d'un unique objet JSON mais ne répare aucune
valeur, clé ou ponctuation et refuse les clés dupliquées, NaN et blocs ambigus.
Le validateur strict du notebook 08 est réutilisé ; confidence doit être un float.
Les catégories de risque et de trafic doivent être cohérentes. Le niveau de risque
des règles est un plancher expérimental, pas une vérité scientifique. Une action
de coordination nécessite au moins un voisin à occupation élevée selon les mêmes
seuils train de 08. Les phases de feux restent interdites faute de contexte de commande.

Le LLM peut produire une justification libre, mais elle doit citer les deux valeurs
d'occupation fournies à six décimales et la phrase : « Incidents inconnus; urgences
inconnues; plan de feux inconnu. » Les autres affirmations incident/urgence/voisin
sont refusées de manière lexicale conservatrice ; les nombres de la justification
doivent correspondre aux occupations fournies ou à l'horizon 15. Ces tests peuvent
rejeter des formulations pourtant légitimes et ne garantissent pas l'absence de
toute hallucination sémantique. Le contrôle d'hallucination porte seulement sur
les raisons JSON analysables ; les réponses non parsées exigent une revue manuelle.
« Reason grounded » signifie conformité à ces
contrôles, pas une annotation humaine démontrant une causalité correcte.

La comparaison rules/LLM est qualitative. L'accord d'action n'est pas une accuracy,
et aucun résultat ne prouve que le LLM est meilleur. Sa confidence est une
auto-évaluation non calibrée, pas une probabilité statistique. Les cinq scénarios
sont les cas illustratifs extrêmes déjà retenus dans 08, pas un échantillon représentatif.
Le mapping location=index NPZ reste hypothétique pour les 25 capteurs aux séries
identiques documentés dans 08 ; les recommandations de voisinage héritent de cette limite.

## Robustesse et architecture

Les tests de JSON invalide, clé manquante, action interdite, confidence >1, faux
incident, modèle indisponible et timeout utilisent des fixtures explicitement
simulées. Ils ne sont pas enregistrés comme sorties du modèle réel.

```text
PEMS08 → Feature Engineering → XGBoost → Prediction +15 min
  → Traffic State Builder (état courant, tendances passées, voisins directs)
  → LLM Decision Agent → JSON et garde-fous → Fallback si nécessaire
  → Recommandation structurée → Orchestrator futur
```

Aucune commande n'est envoyée à des feux ou à un système externe. Les notebooks
01–08, le benchmark et les données sources restent inchangés. L'intégration technique
est réussie si le modèle se charge et génère du texte pour les cinq appels ; le
nombre de décisions acceptées est rapporté séparément, sans masquer les fallbacks.


## Diagnostic du schéma JSON

Un premier essai GGUF utilisait une enveloppe response_format incomplète : le serveur appliquait un JSON générique. Ses sorties sont conservées dans diagnostics/llama_cpp_schema_first_attempt. La correction utilise response_format.json_schema.schema, conformément au code de llama.cpp b11596. Le schéma borne aussi reason à 220 caractères. Les cinq dernières inférences et leurs décisions rejetées restent enregistrées sans modification.
