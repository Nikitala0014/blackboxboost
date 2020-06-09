from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, normalize
import pandas as pd

class MissingValue:
    def __init__(self, data):
        self.data = data

    def impute(self):
        cols_with_missing = [col for col in self.data.columns
                            if self.data[col].isnull().any()]

        if not cols_with_missing:
            return self.data

        s = self.data[cols_with_missing].dtypes == 'object'
        object_cols = list(s[s].index)

        f = self.data[cols_with_missing].dtypes == 'float64'
        float_cols = list(f[f].index)

        i = self.data[cols_with_missing].dtypes == 'int64'
        int_cols = list(i[i].index)

        num_cols = float_cols + int_cols
        
        cols_with_missing_ = num_cols + object_cols
        
        not_missing = self.data.drop(cols_with_missing_, axis=1)

        if num_cols and object_cols:
            mean_imputer = SimpleImputer(strategy='mean')
            imputed_num = pd.DataFrame(mean_imputer.fit_transform(self.data[num_cols]))

            const_imputer = SimpleImputer(strategy='most_frequent')
            imputed_obj = pd.DataFrame(const_imputer.fit_transform(self.data[object_cols]))

            imputed_data = pd.concat([imputed_num, imputed_obj], axis=1)
            finall_data = pd.concat([not_missing, imputed_data], axis=1)

            finall_data.columns = self.data.columns

            return finall_data
        
        if num_cols:
            mean_imputer = SimpleImputer(strategy='mean')
            imputed_num = pd.DataFrame(mean_imputer.fit_transform(self.data[num_cols]))

            finall_data = pd.concat([not_missing, imputed_num], axis=1)
            
        if object_cols:
            const_imputer = SimpleImputer(strategy='most_frequent')
            imputed_obj = pd.DataFrame(const_imputer.fit_transform(self.data[object_cols]))
            
            finall_data = pd.concat([not_missing, imputed_obj], axis=1)

        finall_data.columns = self.data.columns

        return finall_data

class CatVariables:
    def __init__(self, data):
        finall_data = MissingValue(data).impute()
        self.data = finall_data
        

    def oh_cols(self):
        s = self.data.dtypes == 'object'
        object_cols = list(s[s].index)

        if not object_cols:
            return self.data

        oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        oh_cols_data = pd.DataFrame(oh_encoder.fit_transform(self.data[object_cols]))    

        oh_cols_data.index = self.data.index 
        num_data = self.data.drop(object_cols, axis=1)

        finall_data = pd.concat([num_data, oh_cols_data], axis=1)

        return finall_data

class DataPreprocess:
    def __init__(self, data):
        self.data = CatVariables(data).oh_cols()
    
    def normalize(self):
        finall_data = pd.DataFrame(normalize(self.data, norm='l2'))
        finall_data.columns = self.data.columns

        return finall_data