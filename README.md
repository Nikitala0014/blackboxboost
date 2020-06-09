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
You can do Bayesian optimization with initial parameters (obtained using meta-learning) or without.

With meta-learning
------------------
```
import blackboxboost as bbb

# classification
# parameter in process_xgb_clf means the number of epochs of bayesian optimization
meta_params_clf = bbb.BayesOptMeta.BayesOptMeta(train_data_clf, train_labels_clf).process_xgb_clf(100)

# regression
meta_params_reg = bbb.BayesOptMeta.BayesOptMeta(train_data_reg, train_labels_reg).process_xgb_reg(100)
```

You can also get an ensemble of parameters close to the best, which increases efficiency and accuracy.
```
# classification
model_ensemble_clf = bbb.BayesOptMeta.BayesOptMeta(train_data_clf, train_labels_clf).ensemble_of_best_params_xgb_clf(100)
model_ensebmle_clf.fit(train_data_clf, train_labels_clf)

# regression
model_ensemble_reg = bbb.BayesOptMeta.BayesOptMeta(train_data_reg, train_labels_reg).ensemble_of_best_params_xgb_reg(100)
model_ensebmle_reg.fit(train_data_reg, train_labels_reg)
```
