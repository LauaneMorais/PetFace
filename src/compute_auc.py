import pandas as pd
import argparse
import numpy as np
try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        u = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
        return u / (len(pos) * len(neg))


parser = argparse.ArgumentParser()
parser.add_argument('-i',dest='input_csv',required=True)
args = parser.parse_args()

df=pd.read_csv(args.input_csv)
sims=np.array(df['sim'].tolist())
sims=((sims>0)*sims).tolist()
labs=df['label'].tolist()

auc = roc_auc_score(labs, sims)*100
    
print(auc)
