import warnings
import numpy as np
import pandas as pd
import plotnine as gg

# from bsuite.experiments import summary_analysis
# from bsuite.logging import csv_load
# from bsuite.logging import sqlite_load

import analysis as deep_sea_stochastic_analysis
import base_analysis as deep_sea_analysis
import summary_analysis

pd.options.mode.chained_assignment = None
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(12, 8), panel_spacing_x=0.5, panel_spacing_y=0.5)
warnings.filterwarnings('ignore')


# DF = pd.read_csv('results/bsuite_results/0_agent.csv')
# deep_sea_stochastic_df = DF.copy()
# deep_sea_stochastic_plt = deep_sea_stochastic_analysis.find_solution(deep_sea_stochastic_df)
# score = deep_sea_stochastic_analysis.score(deep_sea_stochastic_df)

DF = pd.read_csv('results/bsuite_results/0_agent.csv')
deep_sea_df = DF.copy()
deep_sea_plt = deep_sea_analysis.find_solution(deep_sea_df)
score = deep_sea_analysis.score(deep_sea_df)

data = dict({
          'bsuite_env': 'deep_sea_stochastic',
          'score': score,
          'type': 'deep_sea_stochastic',
          'tags': [None],
          'finished': True,
      })

data = pd.DataFrame(data)

# summary plots
plot = summary_analysis.plot_single_experiment(data, 'deep_sea_stochastic')

# regret plots
regret = deep_sea_stochastic_analysis.plot_regret(deep_sea_plt)

# solve plots
sol = deep_sea_stochastic_analysis.plot_scaling(deep_sea_plt)

print(deep_sea_plt)
print(score)
print(sol)
# print(regret)
# print(plot)
# summary_analysis.plot_single_experiment(BSUITE_SCORE, 'deep_sea_stochastic', SWEEP_VARS)