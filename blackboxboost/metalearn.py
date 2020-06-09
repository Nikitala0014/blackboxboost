import pandas as pd
from blackboxboost.BayesOptMeta import BayesOptMeta
from sklearn.model_selection import train_test_split
from blackboxboost.preprocess_for_meta import DataPreprocess

class MetaLearn:
    def __init__(self, train_data, test_data, epochs, max_evals_train, max_evals_test):
        self.num_tasks_train = len(train_data)
        self.num_tasks_test = len(test_data)
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.max_evals_train = max_evals_train
        self.max_evals_test = max_evals_test
        self.meta_params = {'colsample_bytree': 0.7844773977284409, 'learning_rate': 0.1826993836765384, 'max_depth': 4, 'min_child_weight': 6, 'n_estimators': 45, 'subsample': 0.922357578131324}    
        

    def train(self, task=None):

        for e in range(self.epochs):
            hyperparams_ = []
            for i in range(self.num_tasks_train):
                train_d = self.train_data[i].drop([self.train_data[i].columns[-1]], axis=1)
                train_labels = self.train_data[i].iloc[:, -1:]
                train_data = pd.DataFrame(DataPreprocess(train_d).normalize())
                if task == 'reg':
                    result_params = BayesOptMeta(train_data, train_labels).process_xgb_reg_meta(self.max_evals_train, self.meta_params)
                    hyperparams_.append(result_params)
                elif task == 'clf':
                    result_params = BayesOptMeta(train_data, train_labels).process_xgb_clf_meta(self.max_evals_train, self.meta_params)
                    hyperparams_.append(result_params)
                else:
                    print("Please set the value for the 'task' parameter as 'reg' or 'clf'")
                    return "Please set the value for the 'task' parameter as 'reg' or 'clf'"
        
            dict_test = {}
            for key in hyperparams_[0]:
                result_value = 0
                for result in hyperparams_:
                    result_value += result.get(key)
                if key == 'max_depth' or key == 'min_child_weight' or key == 'n_estimators':
                    result_value = result_value // len(hyperparams_)
                else:
                    result_value = result_value / len(hyperparams_)
                dict_test[key] = result_value
                
            meta_hyper = []
            for i in range(self.num_tasks_test):
                test_d = self.test_data[i].drop([self.test_data[i].columns[-1]], axis=1)
                test_labels = self.test_data[i].iloc[:, -1:]
                test_data = pd.DataFrame(DataPreprocess(test_d).normalize())
                if task == 'reg':
                    result_params_meta = BayesOptMeta(test_data, test_labels).process_xgb_reg_meta(self.max_evals_test, dict_test)
                    meta_hyper.append(result_params_meta)
                elif task == 'clf':
                    result_params_meta = BayesOptMeta(test_data, test_labels).process_xgb_clf_meta(self.max_evals_test, dict_test)
                    meta_hyper.append(result_params_meta)

            m_params = {}
            for key in meta_hyper[0]:
                result_value = 0
                for result in meta_hyper:
                    result_value += result.get(key)
                if key == 'max_depth' or key == 'min_child_weight' or key == 'n_estimators':
                    result_value = result_value // len(meta_hyper)
                else:
                    result_value = result_value / len(meta_hyper)
                m_params[key] = result_value
                    
            self.meta_params = m_params
    
        return self.meta_params