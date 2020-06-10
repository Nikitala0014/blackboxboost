import pandas as pd 
import numpy as np 
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from hyperopt.fmin import generate_trials_to_calculate


#xgb params
xgb_train_params = {
    'n_estimators':     hp.choice('n_estimators', np.arange(50, 1000, dtype=int)),
    'max_depth':        hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
    'learning_rate':    hp.uniform('learning_rate',   0, 0.2),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, dtype=int)),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.8),
    'subsample':        hp.uniform('subsample',        0.8, 1)
}
xgb_fit_params = {
    'early_stopping_rounds': 30,
    'verbose': False
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_train_params
xgb_para['fit_params'] = xgb_fit_params


class BayesOpt:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, train_size=0.7, random_state=1)
        self.x_train = x_train 
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test 


    def process(self, fn_name, space, algo, max_evals):
        fn = getattr(self, fn_name)

        trials = Trials()
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)    
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}
        
        return trials
    
    def process_xgb_reg(self, max_evals):
        trials =  Trials()
        try:
            result = fmin(fn=self.xgb_reg, space=xgb_para, trials=trials, algo=tpe.suggest, max_evals=max_evals)
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}

        return result

    def process_xgb_clf(self, max_evals):
        trials =  Trials()
        try:
            result = fmin(fn=self.xgb_clf, space=xgb_para, trials=trials, algo=tpe.suggest, max_evals=max_evals)
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}

        return result


    def params_to_ensemble(self, fn_name, space, algo, max_evals):
        trials = self.process(fn_name, space, algo, max_evals)
        loss_and_param = []
        for i in trials.trials:
            for k, v in i.items():
                if k == 'result':
                    loss_and_param.append(v)
                    
        best = []
        for i in range(len(loss_and_param)):
            best.append(loss_and_param[i].get('loss'))

        best_loss = min(best)

        params_to_ensemble = []
        for i in range(len(loss_and_param)):
            loss = loss_and_param[i].get('loss')
            if loss <= (best_loss + ((best_loss / 100) * 5)):
                params_to_ensemble.append(loss_and_param[i].get('params'))
                
        return params_to_ensemble


    def ensemble_of_best_params_xgb_reg(self, fn_name, space, algo, max_evals):
        best_params = self.params_to_ensemble(fn_name, space, algo, max_evals)

        models_to_voting = {}
        for i in range(len(best_params)):
            reg = xgb.XGBRegressor(**best_params[i])
            models_to_voting[str(i)] = reg

        model_ensemble = VotingRegressor([(name, model) for name, model in models_to_voting.items()])
        model_ensemble.fit(self.data, self.labels)

        return model_ensemble, best_params


    def ensemble_of_best_params_xgb_clf(self, fn_name, space, algo, max_evals):
        best_params = self.params_to_ensemble(fn_name, space, algo, max_evals)

        models_to_voting = {}
        for i in range(len(best_params)):
            clf = xgb.XGBClassifier(**best_params[i])
            models_to_voting[str(i)] = clf

        model_ensemble = VotingClassifier([(name, model) for name, model in models_to_voting.items()])
        model_ensemble.fit(self.data, self.labels)

        return model_ensemble, best_params
        

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)


    def xgb_clf(self, para):
        clf = xgb.XGBClassifier(**para['reg_params'])
        return self.train_clf(clf, para)


    def train_clf(self, clf, para):
        clf.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = clf.predict(self.x_test)
        loss = log_loss(self.y_test, pred)

        return {'loss': loss, 'params': para['reg_params'], 'status': STATUS_OK}


    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = np.sqrt(mean_squared_error(self.y_test, pred))
        return {'loss': loss, 'params': para['reg_params'], 'status': STATUS_OK}
