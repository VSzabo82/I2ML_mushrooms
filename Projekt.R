# I2ML Projekt #####################################################################################
####################################################################################################

# Preparation --------------------------------------------------------------------------------------
library(tidyverse)
library(mlr3)
library(mlr3viz) # Task Visualisierung
library(mlr3learners) # Learner library
library(mlr3verse)
library(mlr3tuning) # tuning
# library(mlr3filters) # feature importance
# library(ranger)
# library(kknn)
# library(precrec) # Benchmark Plot type "roc" needs this
# library(GGally) # autoplot pairs
# library(paradox) # Parameterset


mushrooms = read.csv("C:/Users/vszab/Desktop/Uni/Statistik/Wahlpflicht/Machine Learning/Projekt/mushrooms.csv")
str(mushrooms)

# veil.type hat nur ein level, sollte entfernt werden
table(mushrooms$veil.type, useNA = "always")
mushrooms = select(mushrooms, -veil.type)
# mushrooms_3 = select(mushrooms, -odor, -gill.size, -gill.color, -ring.type, -spore.print.color, -bruises)
# mushrooms_2 = sample_n(mushrooms, 1000)
summary(mushrooms_2)
# Task ---------------------------------------------------------------------------------------------
task = TaskClassif$new(id = "mushrooms", backend = mushrooms, target = "class", positive = "e")
task$feature_names
autoplot(task)


# Resampling ---------------------------------------------------------------------------------------
resampling = rsmp("holdout", ratio = 0.8)
# resampling = rsmp("cv", folds = 3L)

# Measures -----------------------------------------------------------------------------------------
measures = list(
  msr("classif.auc", id = "auc_train", predict_sets = "train"),
  msr("classif.auc", id = "auc_test")
  , msr("classif.fpr", id = "fpr_test")
)
##   classif.acc, classif.auc, classif.bacc, classif.bbrier,
##   classif.ce, classif.costs, classif.dor, classif.fbeta, classif.fdr,
##   classif.fn, classif.fnr, classif.fomr, classif.fp, classif.fpr,
##   classif.logloss, classif.mbrier, classif.mcc, classif.npv,
##   classif.ppv, classif.precision, classif.recall, classif.sensitivity,
##   classif.specificity, classif.tn, classif.tnr, classif.tp,
##   classif.tpr

# Learner ------------------------------------------------------------------------------------------
learner = list(lrn("classif.featureless", predict_type = "prob", predict_sets = c("train", "test"))
               , lrn("classif.kknn", predict_type = "prob", predict_sets = c("train", "test"))
               , lrn("classif.rpart", predict_type = "prob", predict_sets = c("train", "test"))
               , lrn("classif.ranger", predict_type = "prob", predict_sets = c("train", "test"))
               , lrn("classif.log_reg", predict_type = "prob", predict_sets = c("train", "test"))
               , lrn("classif.naive_bayes", predict_type = "prob", predict_sets = c("train", "test"))
               
               # , lrn("classif.qda", predict_type = "prob")
               # , lrn("classif.lda", predict_type = "prob")
               # Wollten wir nicht benutzen
                )
print(learner)
# Results ------------------------------------------------------------------------------------------
design = benchmark_grid(
  tasks = task,
  learners = learner,
  resamplings = resampling
)
print(design)

bmr <- benchmark(design)
print(bmr)

autoplot(bmr)
autoplot(bmr, type = "roc")
autoplot(bmr, type = "prc")

tab = bmr$aggregate(measures)
print(tab)

# group by levels of task_id, return columns:
# - learner_id
# - rank of col '-auc_train' (per level of learner_id)
# - rank of col '-auc_test' (per level of learner_id)
ranks = tab[, .(learner_id, rank_train = rank(-auc_train), rank_test = rank(-auc_test))]
print(ranks)

####################################################################################################
# tuning -------------------------------------------------------------------------------------------
# Tree, rpart
learner_tree = lrn("classif.rpart", predict_type = "prob")

tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))

measure = msr("classif.ce")

evals20 = term("evals", n_evals = 20)

instance = TuningInstance$new(
  task = task,
  learner = learner_tree,
  resampling = resampling,
  measures = measure,
  param_set = tune_ps,
  terminator = evals20
)
print(instance)

tuner = tnr("grid_search", resolution = 5)
result = tuner$tune(instance)
instance$archive(unnest = "params")[, c("cp", "minsplit", "classif.ce")]

####################################################################################################
# Importance ---------------------------------------------------------------------------------------
# ranger
learner_i = lrn("classif.ranger", predict_type = "prob", importance = "impurity")
# Variable importance mode, one of 'none', 'impurity', 'impurity_corrected', 
# 'permutation'. The 'impurity' measure is the Gini index for classification, 
# the variance of the responses for regression and the sum of test statistics 
# (see splitrule) for survival.

filter = flt("importance", learner = learner_i)
filter$calculate(task)
as.data.table(filter)
