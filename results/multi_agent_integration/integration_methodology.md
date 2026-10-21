# GoFlow — Intégration multi-agent expérimentale

## Périmètre

TrafficPredictionControlAgent, dans src/agents/traffic_prediction_agent.py, assure
la réutilisation des prévisions XGBoost à +15 minutes, la construction d'états
causaux, l'interprétation du risque et une recommandation structurée. Il ne détecte
ni accident ni véhicule d'urgence, ne calcule pas de routes et ne prédit pas les bus.
Les modèles des autres agents ne sont pas implémentés. MockOrchestrator n'est
qu'un outil de test, pas l'orchestrateur final du groupe.

## Contrats

Le message JSON requiert request_id (texte), sensor_id (0..169), timestep (15177..17853)
et traffic_data (trois mesures finies). timestamp peut être null ou une date ISO 8601
avec fuseau, représentant l'enveloppe du message. PEMS08 ne fournit pas ici d'origine
calendaire vérifiable : le timestep est obligatoire et n'est jamais déduit du timestamp.
Les contextes incident/emergency/cooperative/public_transport peuvent être absents,
null ou des objets partiels. Les valeurs absentes restent null, jamais False inventé.
Les flags doivent être de vrais booléens, pas des chaînes. Une gravité/type sans
incident confirmé, ou une priorité sans urgence confirmée, est rejetée.

Le mode courant est un replay historique : les mesures reçues doivent correspondre
au snapshot associé à la prévision sauvegardée. Un voisin externe doit être une
arête directe réelle du graphe, au même timestep, avec mesures concordantes. Une
liste partielle ne supprime pas les autres voisins du graphe. Les données externes
de trafic incompatibles déclenchent une erreur contrôlée, pas une prévision inventée.

La réponse contient le contrat demandé et un objet metadata : timestep, priorité,
trace LLM/fallback, avertissements et marqueur de fixture. En cas d'erreur, forecast,
traffic_state et risk_level sont null, action=manual_review et confidence=0.0 :
ils signalent l'absence de décision valide, pas des mesures ou scores fabriqués.
Les réponses exploitables portent status=warning à cause des limites connues de
mapping et de contexte de feux. Le statut ok est réservé à une future extension
disposant d'informations suffisantes. Le log structuré limite les données affichées.

Les schémas sérialisables INPUT_SCHEMA et OUTPUT_SCHEMA sont dans le module ; la
validation Python couvre aussi les contraintes temporelles et spatiales que les
schémas JSON seuls ne prouvent pas. handle_message intercepte les erreurs et retourne
un contrat error ; les méthodes internes peuvent lever des erreurs explicites.

## Priorité expérimentale et garde-fous

PRIORITY_POLICY : emergency=100, incident=80, severe_congestion=60,
public_transport=40, normal=20. Cette politique est configurable et expérimentale,
pas une règle définitive du système. Les niveaux emergency et incident doivent
rester strictement supérieurs aux autres niveaux pour préserver les garde-fous.
Une urgence présente, ou un incident high/à gravité inconnue, bloque l'appel LLM
et impose manual_review/high. Un deuxième garde-fou vérifie la même priorité après
la couche LLM. Le traitement reste une demande d'arbitrage à l'orchestrateur :
aucune seconde de feu, extension/réduction de phase ni commande réelle n'est émise.
Les autres incidents confirmés et les demandes de priorité transport public sont
aussi renvoyés à une revue orchestrateur, selon leur priorité et le risque existant.
Un contexte externe reçu est attribué à son fournisseur, jamais présenté comme
une détection propre de cet agent. L'authenticité de ces messages relève du futur
orchestrateur ; les tests de contrat ne prouvent pas la fiabilité des autres agents.

## Réutilisation du LLM et des modèles

Aucun modèle n'est réentraîné. predict consulte les 455 090 prévisions existantes,
sans nouvelle inférence XGBoost. Les fonctions et garde-fous de 08 sont réutilisés.
La première intégration est volontairement stricte : risque, état, action et raison
LLM doivent correspondre au contrat déterministe enrichi ; la raison est un texte
de faits autorisé, pas un raisonnement libre sans contrôle.

Sans callable llm_backend injecté, l'agent relit les sorties réelles sauvegardées de 09
sur les cinq états correspondants. metadata.llm_mode indique explicitement un replay,
sans prétendre une nouvelle génération Qwen. Ces sorties restent refusées par les
garde-fous comme dans 09. Pour un état ou contexte sans réponse applicable, le fallback
est utilisé. Un backend local vivant peut être injecté sans changer le contrat ; il
doit recevoir l'état enrichi, fournir un JSON conforme et gérer son délai d'inférence.
Le notebook ne lance pas de serveur ni de nouveau téléchargement. Un backend simulé
valide teste le chemin source=llm, avec integration_test_fixture=true ; il n'est pas
compté comme une véritable décision Qwen ni une preuve de qualité LLM.

## Données, fixtures et vérification

Les mesures utilisées sont les états réels déjà sélectionnés dans 08. Les contextes
positifs incident/emergency/transport, messages volontairement corrompus et backends
simulés sont marqués integration_test_fixture=true. Ils ne sont jamais présentés
comme incidents PEMS08 réels. Le test de garde-fou après LLM est un test unitaire
complémentaire sans échange orchestrateur ; son compte est séparé des messages.

Les vérifications portent sur contrat JSON, conservation de la prévision, erreurs
contrôlées, priorité critique avant/après LLM, voisins réels, contexte manquant,
timeouts et refus de données incohérentes. Il ne s'agit ni d'une accuracy ni d'une
mesure d'efficacité du contrôle routier. La confidence reste non calibrée.
Le mapping location=index NPZ reste une hypothèse pour 25 capteurs aux séries
identiques documentés dans 05/06/08 ; les conseils spatiaux héritent de cette limite.

## Architecture

```text
Incident Detection Agent ──┐
Emergency Priority Agent ──┤
Public Transport Agent ────┤
Cooperative Traffic Agent ─┤
                          ↓
                     Orchestrator
                          ↓
          Traffic Prediction & Control Agent
                          ↓
              XGBoost forecast +15 (sauvegardé)
                          ↓
            LLM + garde-fous déterministes
                          ↓
               recommandation structurée
                          ↓
                     Orchestrator
```

L'interface est prête pour un branchement de test sur l'orchestrateur réel respectant
ce contrat historique. Elle n'est pas un contrôle de feux en production : un flux
temps réel hors de l'export requiert une autre stratégie de forecasting, une
authentification des fournisseurs, un contexte de feux et une politique validée
par l'équipe. Les notebooks 01–09, benchmarks et résultats antérieurs sont préservés.
