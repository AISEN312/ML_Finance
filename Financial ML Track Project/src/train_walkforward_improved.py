import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

try:
    from walkforward import get_index_splits
except Exception:
    from src.walkforward import get_index_splits

try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None


def select_top_k_by_importance(X_train, y_train, feature_names, k=10):
    if lgb is not None:
        clf = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1)
        clf.fit(X_train, y_train)
        imp = clf.feature_importances_
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:k]
    return [feature_names[i] for i in idx]


def get_model(estimator_name='lgb'):
    if estimator_name == 'lgb' and lgb is not None:
        model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
        param_dist = {
            'n_estimators': [100, 200, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63],
            'max_depth': [-1, 3, 6]
        }
    elif estimator_name == 'xgb' and xgb is not None:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_dist = {
            'n_estimators': [100, 200, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9]
        }
    else:
        model = HistGradientBoostingClassifier(random_state=42)
        param_dist = {
            'max_iter': [100, 200, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_leaf_nodes': [15, 31, 63]
        }
    return model, param_dist


def backtest_from_probs(df, prob_col='prob_up', ret_col='ret_1d', threshold=0.5, cost=0.001):
    probs = df[prob_col].values
    rets = df[ret_col].values
    pos = np.zeros_like(probs)
    # long-short policy: long if p>thr, short if p<1-thr, else flat
    for i, p in enumerate(probs):
        if p > threshold:
            pos[i] = 1
        elif p < (1 - threshold):
            pos[i] = -1
        else:
            pos[i] = 0

    prev_pos = 0
    strat_rets = np.zeros_like(rets)
    for i in range(len(rets)):
        cur = pos[i]
        # returns achieved for day i: position at day i * return
        strat_rets[i] = prev_pos * rets[i] - cost * abs(cur - prev_pos)
        prev_pos = cur

    cum = np.cumprod(1 + strat_rets) - 1
    total_return = cum[-1]
    daily_mean = strat_rets.mean()
    daily_std = strat_rets.std(ddof=1) if len(strat_rets) > 1 else 0
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0

    # max drawdown
    peak = np.maximum.accumulate(1 + np.cumsum(strat_rets))
    drawdown = (1 + np.cumsum(strat_rets)) / peak - 1
    max_dd = drawdown.min()

    win_rate = np.mean(strat_rets > 0)
    return dict(total_return=total_return, sharpe=sharpe, max_drawdown=max_dd, win_rate=win_rate, daily_rets=strat_rets)


def main():
    root = Path('.')
    X = pd.read_csv(root / 'data' / 'processed' / 'X_features.csv')
    y = pd.read_csv(root / 'data' / 'processed' / 'y_target.csv')['target']
    dates = pd.read_csv(root / 'data' / 'processed' / 'dates.csv', parse_dates=['date'])['date']

    feature_names = list(X.columns)

    # get splits using dates series
    splits = get_index_splits(dates=pd.to_datetime(dates))

    metrics = []
    preds = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f'Fold {fold_i}: train {len(train_idx)} rows, test {len(test_idx)} rows')
        try:
            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)
            dates_test = dates.iloc[test_idx].reset_index(drop=True)

            # feature selection on train only
            top_feats = select_top_k_by_importance(X_train.values, y_train.values, feature_names, k=10)

            X_train_sel = X_train[top_feats]
            X_test_sel = X_test[top_feats]

            model_name = 'lgb'
            model, param_dist = get_model(model_name)

            pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])

            # wrap param grid for pipeline
            param_grid = {f'clf__{k}': v for k, v in param_dist.items()}

            tscv = TimeSeriesSplit(n_splits=3)
            rs = RandomizedSearchCV(pipe, param_grid, n_iter=10, cv=tscv, scoring='roc_auc', n_jobs=1, random_state=42)
            try:
                rs.fit(X_train_sel, y_train)
                best = rs.best_estimator_
            except Exception as e:
                print('RandomizedSearchCV failed on fold', fold_i, '-', e)
                # fallback: fit simple pipeline without CV search
                pipe.fit(X_train_sel, y_train)
                best = pipe

            probs = best.predict_proba(X_test_sel)[:, 1]

            # metrics
            acc = accuracy_score(y_test, (probs > 0.5).astype(int))
            bacc = balanced_accuracy_score(y_test, (probs > 0.5).astype(int))
            try:
                auc = roc_auc_score(y_test, probs)
            except Exception:
                auc = float('nan')

            metrics.append({'fold': fold_i, 'n_train': len(train_idx), 'n_test': len(test_idx), 'acc': acc, 'bacc': bacc, 'auc': auc})

            # save predictions with date and ret_1d for backtest
            dfp = pd.DataFrame({'date': dates_test, 'fold': fold_i, 'prob_up': probs})
            # attach ret_1d for backtest
            dfp['ret_1d'] = X_test['ret_1d'].values
            preds.append(dfp)
        except Exception as e:
            print('Skipping fold', fold_i, 'due to error:', e)
            continue

    metrics_df = pd.DataFrame(metrics)
    preds_df = pd.concat(preds, ignore_index=True)

    (root / 'data' / 'processed' / 'walkforward_improved_metrics.csv').parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(root / 'data' / 'processed' / 'walkforward_improved_metrics.csv', index=False)
    preds_df.to_csv(root / 'data' / 'processed' / 'walkforward_improved_predictions.csv', index=False)

    # run backtest sweep thresholds
    thresholds = np.arange(0.3, 0.71, 0.05)
    sweep = []
    for thr in thresholds:
        res = backtest_from_probs(preds_df, prob_col='prob_up', ret_col='ret_1d', threshold=thr, cost=0.001)
        sweep.append({'threshold': float(thr), 'total_return': float(res['total_return']), 'sharpe': float(res['sharpe']), 'max_drawdown': float(res['max_drawdown']), 'win_rate': float(res['win_rate'])})

    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(root / 'data' / 'processed' / 'walkforward_improved_threshold_sweep.csv', index=False)

    # pick best threshold by sharpe
    best_thr = float(sweep_df.loc[sweep_df['sharpe'].idxmax(), 'threshold'])
    best_res = backtest_from_probs(preds_df, prob_col='prob_up', ret_col='ret_1d', threshold=best_thr, cost=0.001)

    summary = {'best_threshold': best_thr, 'best_total_return': best_res['total_return'], 'best_sharpe': best_res['sharpe'], 'best_max_drawdown': best_res['max_drawdown'], 'best_win_rate': best_res['win_rate']}
    with open(root / 'data' / 'processed' / 'walkforward_improved_backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('Done; metrics and predictions saved.')


if __name__ == '__main__':
    main()
