from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, normalize
import pandas as pd

class MissingValue:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def impute(self, data):
        cols_with_missing = [col for col in data.columns
                            if data[col].isnull().any()]

        if not cols_with_missing:
            return data

        s = data[cols_with_missing].dtypes == 'object'
        object_cols = list(s[s].index)

        f = data[cols_with_missing].dtypes == 'float64'
        float_cols = list(f[f].index)

        i = data[cols_with_missing].dtypes == 'int64'
        int_cols = list(i[i].index)

        num_cols = float_cols + int_cols
        
        cols_with_missing_ = num_cols + object_cols
        
        not_missing = data.drop(cols_with_missing_, axis=1)

        if num_cols and object_cols:
            mean_imputer = SimpleImputer(strategy='mean')
            imputed_num = pd.DataFrame(mean_imputer.fit_transform(data[num_cols]))

            const_imputer = SimpleImputer(strategy='most_frequent')
            imputed_obj = pd.DataFrame(const_imputer.fit_transform(data[object_cols]))

            imputed_data = pd.concat([imputed_num, imputed_obj], axis=1)
            finall_data = pd.concat([not_missing, imputed_data], axis=1)

            finall_data.columns = data.columns

            return finall_data
        
        if num_cols:
            mean_imputer = SimpleImputer(strategy='mean')
            imputed_num = pd.DataFrame(mean_imputer.fit_transform(data[num_cols]))

            finall_data = pd.concat([not_missing, imputed_num], axis=1)
            
        if object_cols:
            const_imputer = SimpleImputer(strategy='most_frequent')
            imputed_obj = pd.DataFrame(const_imputer.fit_transform(data[object_cols]))
            
            finall_data = pd.concat([not_missing, imputed_obj], axis=1)

        finall_data.columns = data.columns

        return finall_data

    def imputer_data(self):
        train_imputed = self.impute(self.train_data)
        test_imputed = self.impute(self.test_data)

        return train_imputed, test_imputed


class CatVariables:
    def __init__(self, train_data, test_data):
        train_imputed, test_imputed = MissingValue(train_data, test_data).imputer_data()
        self.train_data = train_imputed
        self.test_data = test_imputed
        

    def oh_cols(self):
        s = self.train_data.dtypes == 'object'
        object_cols = list(s[s].index)

        c = self.test_data.dtypes == 'object'
        object_cols_test = list(c[c].index)

        oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(self.train_data[object_cols]))
        oh_cols_test = pd.DataFrame(oh_encoder.transform(self.test_data[object_cols_test]))
        

        oh_cols_train.index = self.train_data.index 
        num_data = self.train_data.drop(object_cols, axis=1)

        oh_cols_test.index = self.test_data.index 
        num_data_test = self.test_data.drop(object_cols_test, axis=1)

        right_train = pd.concat([num_data, oh_cols_train], axis=1)
        right_test = pd.concat([num_data_test, oh_cols_test], axis=1)

        return right_train, right_test

class DataPreprocess:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
    
    def normalize(self):
        right_train, right_test = CatVariables(self.train_data, self.test_data).oh_cols()
        train_normalized = pd.DataFrame(normalize(right_train, norm='l2'))
        test_normalized = pd.DataFrame(normalize(right_test, norm='l2'))

        train_normalized.columns = right_train.columns
        test_normalized.columns = right_test.columns

        return train_normalized, test_normalized