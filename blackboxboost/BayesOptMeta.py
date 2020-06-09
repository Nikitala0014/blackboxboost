from blackboxboost.preprocess_for_meta import DataPreprocess
from blackboxboost.BayesOpt import xgb_para
import pandas as pd 
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import VotingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from hyperopt.fmin import generate_trials_to_calculate
import xgboost as xgb 
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from pymfe.mfe import MFE
import math
import pickle

class BayesOptMeta:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, train_size=0.7, random_state=1)
        self.x_train = x_train 
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def process_meta(self, fn_name, space, algo, max_evals):
        fn = getattr(self, fn_name)
        if fn_name == 'xgb_reg':
            trials = generate_trials_to_calculate([self.meta_param_xgb_reg()])
        else:
            trials = generate_trials_to_calculate([self.meta_param_xgb_clf()])

        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)    
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}
        
        return result
    
    def process_xgb_reg_meta(self, max_evals, meta_params):
        trials =  generate_trials_to_calculate([meta_params])
        try:
            result = fmin(fn=self.xgb_reg, space=xgb_para, trials=trials, algo=tpe.suggest, max_evals=max_evals)
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}

        return result

    def process_xgb_reg(self, max_evals):
        trials =  generate_trials_to_calculate([self.meta_param_xgb_reg()])
        try:
            result = fmin(fn=self.xgb_reg, space=xgb_para, trials=trials, algo=tpe.suggest, max_evals=max_evals)
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}

        return result

    def process_xgb_clf_meta(self, max_evals, meta_params):
        trials =  generate_trials_to_calculate([meta_params])
        try:
            result = fmin(fn=self.xgb_clf, space=xgb_para, trials=trials, algo=tpe.suggest, max_evals=max_evals)
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}

        return result

    def process_xgb_clf(self, max_evals):
        trials =  generate_trials_to_calculate([self.meta_param_xgb_clf()])
        try:
            result = fmin(fn=self.xgb_clf, space=xgb_para, trials=trials, algo=tpe.suggest, max_evals=max_evals)
        except Exception as e:
            return {'status': STATUS_FAIL, 'exception': str(e)}

        return result

    def meta_param_xgb_clf(self):
        data = pd.concat([self.data, self.labels], axis=1)
        data_ = pd.DataFrame(normalize(data))
        data_.columns = data.columns
        x = data_.drop([data_.columns[-1]], axis=1)
        y = data_.iloc[:, -1:]
        if len(data) > 3000:
            x = x.iloc[:3000]
            y = y.iloc[:3000]
        mfe = MFE(groups=['general', 'statistical'])
        mfe.fit(x.values, y.values)
        ft = mfe.extract()

        feature_dict = {x:y for x, y in zip(ft[0], ft[1])}

        for k, v in feature_dict.items():
            if math.isnan(v):
                feature_dict[k] = 0.00001
        
        feature_data = []
        for k, v in feature_dict.items():
            feature_data.append(v)

        feature_ = csr_matrix(feature_data)

        kmeans_clf = pickle.load(open('/Users/Nikita/Desktop/classification datasets/kmeans_clf.pkl', 'rb'))

        meta_params_0_clf = {'subsample': 0.85871509130206, 'n_estimators': 548, 'colsample_bytree': 0.6159585946755198, 'max_depth': 6, 'learning_rate': 0.1519163133871961, 'min_child_weight': 0}
        meta_params_1_clf = {'subsample': 0.8910152781465692, 'n_estimators': 264, 'colsample_bytree': 0.5534115708933309, 'max_depth': 9, 'learning_rate': 0.10132128433332305, 'min_child_weight': 0}    
        meta_params_2_clf = {'min_child_weight': 0, 'subsample': 0.909269885479936, 'n_estimators': 309, 'learning_rate': 0.19992694541533446, 'max_depth': 9, 'colsample_bytree': 0.40056370089015014}
 
        pred = kmeans_clf.predict(feature_)
        if pred == 0:
            return meta_params_0_clf
        elif pred == 1:
            return meta_params_1_clf
        elif pred == 2:
            return meta_params_2_clf

    def meta_param_xgb_reg(self):
        data = pd.concat([self.data, self.labels], axis=1)
        data_ = pd.DataFrame(normalize(data))
        data_.columns = data.columns
        x = data_.drop([data_.columns[-1]], axis=1)
        y = data_.iloc[:, -1:]
        if len(data) > 3000:
            x = x.iloc[:3000]
            y = y.iloc[:3000]
        mfe = MFE(groups=['general', 'statistical'])
        mfe.fit(x.values, y.values)
        ft = mfe.extract()

        feature_dict = {x:y for x, y in zip(ft[0], ft[1])}

        for k, v in feature_dict.items():
            if math.isnan(v):
                feature_dict[k] = 0.00001
        
        feature_data = []
        for k, v in feature_dict.items():
            feature_data.append(v)

        feature_ = csr_matrix(feature_data)

        kmeans_reg = pickle.load(open('/Users/Nikita/Desktop/reg_openml/model_reg.pkl', 'rb'))

        meta_params_0_reg = {'n_estimators': 547, 'learning_rate': 0.14502110379186764, 'subsample': 0.8802447552806882, 'min_child_weight': 2, 'max_depth': 5, 'colsample_bytree': 0.6338079055523455}
        meta_params_1_reg = {'n_estimators': 482, 'learning_rate': 0.11999175898448614, 'subsample': 0.8842506424173399, 'min_child_weight': 3, 'max_depth': 3, 'colsample_bytree': 0.6768310665291043}
        meta_params_2_reg = {'n_estimators': 508, 'learning_rate': 0.1188653684155892, 'subsample': 0.8901090475427362, 'min_child_weight': 2, 'max_depth': 7, 'colsample_bytree': 0.5238370048139117}

        pred = kmeans_reg.predict(feature_)
        if pred == 0:
            return meta_params_0_reg
        elif pred == 1:
            return meta_params_1_reg
        elif pred == 2:
            return meta_params_2_reg


    def params_to_ensemble(self, fn_name, space, algo, max_evals):
        _, trials = self.process_meta(fn_name, space, algo, max_evals)
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


    def ensemble_of_best_params_xgb_reg(self, max_evals):
        best_params = self.params_to_ensemble(fn_name='xgb_reg', space=xgb_para, algo=tpe.suggest, max_evals=max_evals)

        models_to_voting = {}
        for i in range(len(best_params)):
            reg = xgb.XGBRegressor(**best_params[i])
            models_to_voting[str(i)] = reg

        model_ensemble = VotingRegressor([(name, model) for name, model in models_to_voting.items()])

        return model_ensemble

    def ensemble_of_best_params_xgb_clf(self, max_evals):
        best_params = self.params_to_ensemble(fn='xgb_clf', space=xgb_para, trials=trials, algo=tpe.suggest, max_evals=max_evals)

        models_to_voting = {}
        for i in range(len(best_params)):
            clf = xgb.XGBClassifier(**best_params[i])
            models_to_voting[str(i)] = clf

        model_ensemble = VotingClassifier([(name, model) for name, model in models_to_voting.items()])

        return model_ensemble


    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def xgb_clf(self, para):
        clf = xgb.XGBClassifier(**para['reg_params'])
        return self.train_clf(clf, para)


    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = np.sqrt(mean_squared_error(self.y_test, pred))
        return {'loss': loss, 'params': para['reg_params'], 'status': STATUS_OK}

    def train_clf(self, clf, para):
        clf.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = clf.predict(self.x_test)
        loss = log_loss(self.y_test, pred)

        return {'loss': loss, 'params': para['reg_params'], 'status': STATUS_OK}
        