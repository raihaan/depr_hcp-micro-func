import sys, os
import numpy as np
import pandas as pd

restr = pd.read_csv(sys.argv[1])
unrestr = pd.read_csv(sys.argv[2])

merged = restr.merge(unrestr[['Subject','Gender']], how='inner',on='Subject')
merged_sorted = merged.sort_values(by='Age_in_Yrs')

merged_sorted.to_csv(sys.argv[3],index=False)
