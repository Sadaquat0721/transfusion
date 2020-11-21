{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Recency (months)</th>\n",
       "      <th>Frequency (times)</th>\n",
       "      <th>Monetary (c.c. blood)</th>\n",
       "      <th>Time (months)</th>\n",
       "      <th>whether he/she donated blood in March 2007</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>50</td>\n",
       "      <td>12500</td>\n",
       "      <td>98</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>3250</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>16</td>\n",
       "      <td>4000</td>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>20</td>\n",
       "      <td>5000</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>6000</td>\n",
       "      <td>77</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Recency (months)  Frequency (times)  Monetary (c.c. blood)  Time (months)  \\\n",
       "0                 2                 50                  12500             98   \n",
       "1                 0                 13                   3250             28   \n",
       "2                 1                 16                   4000             35   \n",
       "3                 2                 20                   5000             45   \n",
       "4                 1                 24                   6000             77   \n",
       "\n",
       "   whether he/she donated blood in March 2007  \n",
       "0                                           1  \n",
       "1                                           1  \n",
       "2                                           1  \n",
       "3                                           1  \n",
       "4                                           0  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "transfusion=pd.read_csv('transfusion.csv.txt')\n",
    "transfusion.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 748 entries, 0 to 747\n",
      "Data columns (total 5 columns):\n",
      " #   Column                                      Non-Null Count  Dtype\n",
      "---  ------                                      --------------  -----\n",
      " 0   Recency (months)                            748 non-null    int64\n",
      " 1   Frequency (times)                           748 non-null    int64\n",
      " 2   Monetary (c.c. blood)                       748 non-null    int64\n",
      " 3   Time (months)                               748 non-null    int64\n",
      " 4   whether he/she donated blood in March 2007  748 non-null    int64\n",
      "dtypes: int64(5)\n",
      "memory usage: 29.3 KB\n"
     ]
    }
   ],
   "source": [
    "transfusion.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "transfusion.rename(columns={'whether he/she donated blood in March 2007':'target'},inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Recency (months)</th>\n",
       "      <th>Frequency (times)</th>\n",
       "      <th>Monetary (c.c. blood)</th>\n",
       "      <th>Time (months)</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>50</td>\n",
       "      <td>12500</td>\n",
       "      <td>98</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>3250</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Recency (months)  Frequency (times)  Monetary (c.c. blood)  Time (months)  \\\n",
       "0                 2                 50                  12500             98   \n",
       "1                 0                 13                   3250             28   \n",
       "\n",
       "   target  \n",
       "0       1  \n",
       "1       1  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "transfusion.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.762\n",
       "1    0.238\n",
       "Name: target, dtype: float64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "transfusion.target.value_counts(normalize=True).round(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    " from sklearn.model_selection import train_test_split\n",
    "   # X_train,X_test,y_train,y_test=train_test_split(transfusion.drop(columns='target'),transfusion.target,test_size=0.25,random_states=42,stratify=transfusion.target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test=train_test_split(transfusion.drop(columns='target'),transfusion.target,test_size=0.25,random_state=42,stratify=transfusion.target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Recency (months)</th>\n",
       "      <th>Frequency (times)</th>\n",
       "      <th>Monetary (c.c. blood)</th>\n",
       "      <th>Time (months)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>334</th>\n",
       "      <td>16</td>\n",
       "      <td>2</td>\n",
       "      <td>500</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99</th>\n",
       "      <td>5</td>\n",
       "      <td>7</td>\n",
       "      <td>1750</td>\n",
       "      <td>26</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Recency (months)  Frequency (times)  Monetary (c.c. blood)  Time (months)\n",
       "334                16                  2                    500             16\n",
       "99                  5                  7                   1750             26"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "#from tpot import TPOTClassifier\n",
    "from sklearn.metrics import roc_auc_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Sadaquat Hussain\\anaconda3\\lib\\site-packages\\tpot\\builtins\\__init__.py:36: UserWarning: Warning: optional dependency `torch` is not available. - skipping import of NN models.\n",
      "  warnings.warn(\"Warning: optional dependency `torch` is not available. - skipping import of NN models.\")\n"
     ]
    }
   ],
   "source": [
    "from tpot import TPOTClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(FloatProgress(value=0.0, description='Optimization Progress', style=ProgressStyle(description_w…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Generation 1 - Current best internal CV score: 0.7422459184429089\n",
      "\n",
      "Generation 2 - Current best internal CV score: 0.7422459184429089\n",
      "\n",
      "Generation 3 - Current best internal CV score: 0.7422459184429089\n",
      "\n",
      "Generation 4 - Current best internal CV score: 0.7422459184429089\n",
      "\n",
      "Best pipeline: LogisticRegression(input_matrix, C=25.0, dual=False, penalty=l2)\n",
      "\n",
      "AUC scoren: 0.7858\n",
      "\n",
      "Best pipeline steps:\n",
      "1.LogisticRegression(C=25.0, random_state=42)\n"
     ]
    }
   ],
   "source": [
    "tpot=TPOTClassifier(generations=4,population_size=20,verbosity=2,scoring='roc_auc',random_state=42,disable_update_check=True,config_dict='TPOT light')\n",
    "tpot.fit(X_train,y_train)\n",
    "tpot_auc_score=roc_auc_score(y_test,tpot.predict_proba(X_test)[:,1]).round(4)\n",
    "print(f'\\nAUC scoren: {tpot_auc_score:.4f}')\n",
    "print('\\nBest pipeline steps:',end='\\n')\n",
    "for idx,(name,transform) in enumerate(tpot.fitted_pipeline_.steps,start=1):\n",
    "    print(f'{idx}.{transform}')\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recency (months)              66.929\n",
      "Frequency (times)             33.830\n",
      "Monetary (c.c. blood)    2114363.700\n",
      "Time (months)                611.147\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "print(X_train.var().round(3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recency (months)      66.929\n",
      "Frequency (times)     33.830\n",
      "Time (months)        611.147\n",
      "monetary_log           0.837\n"
     ]
    }
   ],
   "source": [
    "X_train_normed,X_test_normed=X_train.copy(),X_test.copy()\n",
    "col_to_normalize=X_train_normed.var().idxmax(axis=1)\n",
    "for df_ in [X_train_normed,X_test_normed]:\n",
    "    df_['monetary_log']=np.log(df_[col_to_normalize])\n",
    "    df_.drop(columns=col_to_normalize,inplace=True)\n",
    "print(X_train_normed.var().round(3).to_string())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AUC score:0.7891\n"
     ]
    }
   ],
   "source": [
    "from sklearn import linear_model\n",
    "logreg=linear_model.LogisticRegression(solver='liblinear',random_state=42)\n",
    "logreg.fit(X_train_normed,y_train)\n",
    "logreg_auc_score=roc_auc_score(y_test,logreg.predict_proba(X_test_normed)[:,1])\n",
    "print(f'AUC score:{logreg_auc_score:0.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('logreg', 0.7891), ('tpot', 0.7858)]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from operator import itemgetter\n",
    "sorted([('tpot',tpot_auc_score.round(4)),('logreg',logreg_auc_score.round(4))],key=itemgetter(1),reverse=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
