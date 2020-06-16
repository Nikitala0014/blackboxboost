blackboxboost
===============

Library for machine learning automation based on gradient boosting, meta-learning, 
bayesian optimization, as well as an ensemble of the best parameters obtained in the bayesian optimization.

Getting started
===============

Install blackboxboost from PyPI
```
& pip install blackboxboost
```
Data preprocessing
------------------
```
import blackboxboost as bbb
```
Use this if the test data is independent of the training data, for example, you do not need to pass the 
test data to the algorithm as in the kaggle, because there you need the same number of columns:
```
# Works with missing and categorical values and also normalizes.
# You must use this or your methods before passing to the methods clf or reg below.
# Use this for Bayesian optimization without meta-learning (BayesOpt)
train_data = bbb.preprocess_for_meta.DataPreprocess(train_).normalize()

# Without normalization, only categorical and missing values.
# Use this for Bayesian optimization with meta-learning (BayesOptMeta)
train_data = bbb.preprocess_for_meta.CatVariables(train_).oh_cols()
```
Use this if you have test data for which you need to build and train a model:
```
# Works with missing and categorical values and also normalizes.
right_train, right_test = bbb.preprocess_data.DataPreprocess(train_data, test_data).normalize()

# Without normalization, only categorical and missing values.
right_train, right_test = bbb.preprocess_data.CatVariables(train_data, test_data).oh_cols()
```
You can do Bayesian optimization with initial parameters (obtained using meta-learning) or without.
======================
With meta-learning
------------------
```
# classification
# Parameter in process_xgb_clf means the number of epochs of bayesian optimization.
meta_params_clf = bbb.BayesOptMeta.BayesOptMeta(train_data_clf, train_labels_clf).process_xgb_clf(100)

# regression
meta_params_reg = bbb.BayesOptMeta.BayesOptMeta(train_data_reg, train_labels_reg).process_xgb_reg(100)
```

You can also get an ensemble of parameters close to the best, which increases efficiency and accuracy:
```
# classification
model_ensemble_clf, best_params = bbb.BayesOptMeta.BayesOptMeta(train_data_clf, train_labels_clf).ensemble_of_best_params_xgb_clf(100)
model_ensebmle_clf.fit(train_data_clf, train_labels_clf)
print(best_params)

# regression
model_ensemble_reg, best_params = bbb.BayesOptMeta.BayesOptMeta(train_data_reg, train_labels_reg).ensemble_of_best_params_xgb_reg(100)
model_ensebmle_reg.fit(train_data_reg, train_labels_reg)
print(best_params)
```
Without meta-learning
---------------------

```
# classification
meta_params = bbb.BayesOpt.BayesOpt(train_data_clf, train_labels_clf).process_xgb_clf(100)

# regression
meta_params = bbb.BayesOpt.BayesOpt(train_data_clf, train_labels_reg).process_xgb_reg(100)
```
Ensemble of parameters close to the best:
```
# classification
model_ensemble_clf, best_params = bbb.BayesOpt.BayesOpt(train_data_clf, train_labels_clf).ensemble_of_best_params_xgb_clf(100)
# The model is already trained
predict_clf = model_ensemble_clf.predict(test_data_clf)
print(best_params)

# regression
model_ensemble_reg, best_params = bbb.BayesOpt.BayesOpt(train_data_reg, train_labels_reg).ensemble_of_best_params_xgb_reg(100)
predict_reg = model_ensemble_reg.predict(test_data_reg)
print(best_params)
```
Meta-Learning
-------------
If you want to try the meta-learning algorithm for your tasks, then you need two training and test lists, there should be data sets similar to yours for which you will use hyperparameters. Set target values to last column.
This algorithm is based on the MAML algorithm and is optimized for hyperparameters while the MAML is built for weights.
```
#You must set the task parameter as 'reg' or 'clf'.
meta_learn_params = bbb.metalearn.MetaLearn(train_data, test_data, epochs, max_evals_train, max_evals_test).train(task=None)
```
Future work
===========
This repository, this library is just a test version, the first step on my part is to automate machine learning, in order to make machine learning an affordable tool for all developers.
The next steps will be towards deep learning and meta-learning, not only for hyperparameters, but also for weights and models.
