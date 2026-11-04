"""Construit et exécute notebooks/01_data_understanding.ipynb (figures copiées dans reports/figures)."""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# 01 — Data Understanding\n\n"
       "**Question métier :** combien de temps les véhicules d'urgence perdent-ils aux feux aujourd'hui, "
       "et qu'est-ce qui explique ces pertes ?\n\n"
       "Les données sont produites par SUMO avec le **système actuel** : feux à cycle fixe de 90 s, "
       "sans aucune priorité. 30 scénarios aléatoires sur le carrefour isolé + 6 scénarios sur le corridor de 3 carrefours. "
       "Régénérer : `python -m data.export_dataset`."),
    code("import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\n"
         "from pathlib import Path\n"
         "RAW = Path('../data/raw'); FIG = Path('../reports/figures'); FIG.mkdir(parents=True, exist_ok=True)\n"
         "sns.set_theme(style='whitegrid', palette='deep'); plt.rcParams['figure.dpi'] = 110\n"
         "tables = {n: pd.read_csv(RAW / f'{n}.csv') for n in ['scenarios','lanes','trips','priority_tracks','priority_requests','signals']}"),
    md("## 1. Inventaire des tables"),
    code("pd.DataFrame([{'table': n, 'lignes': len(d), 'colonnes': d.shape[1], 'valeurs manquantes': int(d.isna().sum().sum()),\n"
         "               'colonnes (extrait)': ', '.join(d.columns[:7])} for n, d in tables.items()])"),
    code("tables['scenarios'].groupby('network').agg(scenarios=('scenario','count'), demande_moy=('demand_level','mean'),\n"
         "    urgences=('n_priority','sum')).round(2)"),
    md("Les valeurs manquantes de `priority_tracks` correspondent aux véhicules qui ont déjà franchi leur dernier feu "
       "(`next_tls` vide) : elles sont **structurelles**, pas des erreurs."),
    md("## 2. Qualité des données"),
    code("trips = tables['trips']\n"
         "checks = {\n"
         "  'durées négatives': int((trips.duration < 0).sum()),\n"
         "  'temps perdu > durée': int((trips.time_loss > trips.duration + 1).sum()),\n"
         "  'doublons (scénario, véhicule)': int(trips.duplicated(['scenario','vehicle_id']).sum()),\n"
         "  'vitesses négatives (voies)': int((tables['lanes'].mean_speed < 0).sum()),\n"
         "  'états de feux distincts': tables['signals'].state.nunique()}\n"
         "checks"),
    code("trips.describe().round(1)"),
    md("## 3. Temps perdu par type de véhicule (système actuel)"),
    code("order = ['car','bus_transit','ambulance','fire_truck','police']\n"
         "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
         "sns.boxplot(data=trips, x='vtype', y='time_loss', order=order, ax=ax[0], showfliers=False)\n"
         "ax[0].set(title='Temps perdu par trajet (s)', xlabel='', ylabel='secondes')\n"
         "trips.groupby('vtype').stops.mean().reindex(order).plot.bar(ax=ax[1], color=sns.color_palette()[1], rot=0)\n"
         "ax[1].set(title='Arrêts moyens par trajet', xlabel='', ylabel='arrêts')\n"
         "plt.tight_layout(); plt.savefig(FIG / 'du_time_loss_by_type.png'); plt.show()\n"
         "trips.groupby('vtype')[['time_loss','waiting_time','stops']].mean().reindex(order).round(1)"),
    md("**Constat clé :** avec des feux à cycle fixe, une ambulance perd autant de temps qu'une voiture ordinaire "
       "(≈ 27 s) et s'arrête presque une fois par trajet. Les feux ne savent pas qu'elle est prioritaire."),
    md("## 4. Exemple : une ambulance bloquée au rouge"),
    code("tr = tables['priority_tracks']; sig = tables['signals']\n"
         "amb = tr[(tr.scenario=='S2_urgence_0') & (tr.vehicle_id=='amb_0')]\n"
         "fig, ax = plt.subplots(figsize=(11, 3.6))\n"
         "ax.plot(amb.time, amb.speed * 3.6, color='crimson', lw=2, label='vitesse ambulance (km/h)')\n"
         "for tls, col in zip(['J1','J2','J3'], sns.color_palette('muted', 3)):\n"
         "    near = amb[(amb.next_tls == tls) & (amb.dist_to_tls < 60)]\n"
         "    if len(near): ax.axvspan(near.time.min(), near.time.max(), color=col, alpha=.2, label=f'approche de {tls} (< 60 m)')\n"
         "ax.set(xlabel='temps de simulation (s)', ylabel='km/h', title='Ambulance amb_0 sur le corridor — cycle fixe')\n"
         "ax.legend(loc='lower right', fontsize=8); plt.tight_layout(); plt.savefig(FIG / 'du_ambulance_speed.png'); plt.show()"),
    md("La vitesse tombe à zéro à l'approche d'au moins un carrefour : c'est ce temps d'arrêt que le système doit supprimer."),
    md("## 5. Files d'attente dans le temps (corridor, congestion)"),
    code("lanes = tables['lanes']\n"
         "q = lanes[lanes.scenario=='S3_congestion_0'].groupby(['time','tls_id']).halting.sum().unstack()\n"
         "ax = q.rolling(6).mean().plot(figsize=(11, 3.6), lw=1.6)\n"
         "ax.set(title='Véhicules arrêtés par carrefour — S3 congestion, cycle fixe', xlabel='temps (s)', ylabel='véhicules arrêtés')\n"
         "plt.tight_layout(); plt.savefig(FIG / 'du_queues_corridor.png'); plt.show()"),
    md("## 6. Demande et attente"),
    code("tri = trips.merge(tables['scenarios'], on='scenario')\n"
         "agg = tri[tri.vtype=='car'].groupby(['scenario','network','demand_level']).waiting_time.mean().reset_index()\n"
         "ax = sns.scatterplot(data=agg, x='demand_level', y='waiting_time', hue='network', s=60)\n"
         "ax.set(title='Attente moyenne des voitures selon le niveau de demande', xlabel='niveau de demande', ylabel='attente moyenne (s)')\n"
         "plt.tight_layout(); plt.savefig(FIG / 'du_demand_vs_wait.png'); plt.show()\n"
         "print('corrélation (Spearman) demande / attente :', round(agg.demand_level.corr(agg.waiting_time, method='spearman'), 2))"),
    md("## 7. Répartition des files par approche (carrefour isolé)"),
    code("single = lanes[lanes.tls_id=='C']\n"
         "pv = single.groupby(['approach']).agg(arrets_moy=('halting','mean'), occupation_moy=('occupancy','mean'), vitesse_moy=('mean_speed','mean'))\n"
         "pv.round(2)"),
    md("## 8. Demandes de priorité (V2I)"),
    code("rq = tables['priority_requests']\n"
         "rq['statut'] = np.select([rq.valid_request, ~rq.siren_on & (rq.priority_class!='TRANSIT'), ~rq.confirmed],\n"
         "                         ['valide', 'sirène éteinte', 'non confirmée'], 'bus à l\\'heure')\n"
         "tab = pd.crosstab(rq.priority_class, rq.statut); tab"),
    code("ax = tab.plot.barh(stacked=True, figsize=(9, 3.2))\n"
         "ax.set(title='Demandes de priorité par classe et statut', xlabel='demandes', ylabel='')\n"
         "plt.tight_layout(); plt.savefig(FIG / 'du_requests.png'); plt.show()"),
    md("## 9. Conclusions pour la modélisation\n\n"
       "1. **Problème confirmé :** au cycle fixe, les urgences perdent ≈ autant de temps que le trafic ordinaire.\n"
       "2. **Variables utiles** pour l'état de PPO : files et densités par voie (elles expliquent l'attente), "
       "phase courante et durée, occupation en aval, et le contexte de priorité (approche, ETA).\n"
       "3. **Toutes les demandes ne sont pas valables :** sirènes éteintes, demandes non confirmées, bus à l'heure. "
       "Le système doit filtrer avant d'agir (rôle du coordinateur et du Safety Controller).\n"
       "4. **Forte variabilité selon la demande :** l'entraînement doit couvrir un large intervalle de niveaux de trafic, "
       "et l'évaluation doit se faire sur des graines non vues."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, HERE / "01_data_understanding.ipynb")
print("ok")
