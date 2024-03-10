source(file.path("evalUtils.r", fsep = .Platform$file.sep))
source(file.path("MixRF.R", fsep = .Platform$file.sep))

# Environment preparation
library(readr)
library(mefa)
library(lme4)
library(lmerTest)
library(Hmisc)
library(car)
library(sjPlot)
library(optimx)
library(MuMIn)
library(boot)
library(plyr)
library(doParallel)
library(caret)
library(ranger)
library(ROCR)
library(data.table)
library(leaps)

registerDoParallel(cores = detectCores())

# Data preparation
context <- read.csv(file.path("data","context.csv", fsep = .Platform$file.sep), row.names = 1)
project_names <- row.names(context)
metrics <- c("ns","nd","nf","la","ld","lt","norm_entropy")

projects <- list()
for (i in 1:length(project_names)) {
  projects[[i]] <- read_csv(file.path("data",paste(project_names[i], ".csv", sep = ''), 
                                      fsep = .Platform$file.sep),
                            col_types = cols(contains_bug = col_logical(), 
                                             fix = col_logical()))
  
  projects[[i]]$loc <- projects[[i]]$la + projects[[i]]$ld
  
  projects[[i]]$norm_entropy <- 0
  tmp_norm_entropy <- projects[[i]]$entrophy / sapply(projects[[i]]$nf, log2) # Normalize entropy
  projects[[i]][projects[[i]]$nf >= 2, "norm_entropy"] <- tmp_norm_entropy[projects[[i]]$nf >= 2]
  
  projects[[i]]$project <- project_names[i]
  
  projects[[i]] <- cbind(projects[[i]], rep(context[project_names[i],], times = nrow(projects[[i]])))
} 


scale_metrics <- c("ns","nf","lt","norm_entropy","relative_churn")

for (i in 1:length(project_names)) {
  projects[[i]]$relative_churn <- 0
  tmp_relative_churn <- (projects[[i]]$la + projects[[i]]$ld) / projects[[i]]$lt # (la+ld)/lt
  projects[[i]][projects[[i]]$lt >= 1, "relative_churn"] <- tmp_relative_churn[projects[[i]]$lt >= 1]
  projects[[i]][is.na(projects[[i]])] <- 0
  projects[[i]][,scale_metrics] <- lapply(projects[[i]][,scale_metrics], scale) # Data scaling
}

all_projects <- projects[[1]]
for (i in 2:length(project_names)) {
  all_projects <- rbind(all_projects, projects[[i]])
} 
all_projects$project <- as.factor(all_projects$project) # Merged dataset

all_projects.bglm <- all_projects[, c("fix", "ns", "nf", "norm_entropy", "relative_churn", "lt", "nlanguage", "loc", "nfile", "ncommit", "ndev", "linked", "age", "exp", "rexp", "sexp", "nuc", "contains_bug")]
rename(all_projects.bglm, y=contains_bug)

best.logit <- bestglm(all_projects.bglm,
                IC = "AIC",                
                family=binomial,
                method = "exhaustive")

summary(best.logit$BestModel)
best.logit$Subsets

print(r.squaredGLMM(best.logit$BestModel))

# bestsub.model <- regsubsets(contains_bug ~ fix + ns +linked+ nf +ndev+age+ norm_entropy + relative_churn + lt +exp+rexp+sexp+nuc, data = all_projects, nvmax = 14)

# summary(bestsub.model)

#  cbind( 
#     Cp     = summary(bestsub.model)$cp,
#     r2     = summary(bestsub.model)$rsq,
#     Adj_r2 = summary(bestsub.model)$adjr2,
#     BIC    =summary(bestsub.model)$bic
# )

# # ===================== RQ2 ========================
# # Model training
# print("RQ2")
# global.jit <- glm(contains_bug ~ fix + ns + nd+ nf + norm_entropy + age, data = all_projects, family = "binomial")


# # Median Absolute Error for entropy and intercepts
# global.jit.entropy <- coef(global.jit)[5]
# global.jit.intercept <- coef(global.jit)[1]
# # median(abs(local.entropy - global.jit.entropy)) # MAE for Entropy
# # median(abs(local.intercept - global.jit.intercept)) # MAE for intercepts

# # Chisq
# Anova(global.jit, Type = 2)

# # Goodness-of-fit
# print(r.squaredGLMM(global.jit))

# ===================== RQ3 ========================
print("RQ3")

# # Performance of Global JIT Model (Logistic Regression)
# print("Running Global")
# get_lr_perf <- function(i) {
#   training_set <- subset(all_projects, all_projects$project != project_names[i])
#   testing_set <- subset(all_projects, all_projects$project == project_names[i])
  
#   tmp_lr_model <- glm(contains_bug ~  fix + ns + nd+ nf + norm_entropy + age, data = training_set, family="binomial")
#   tmp_lr_pred <- predict(tmp_lr_model, newdata = testing_set, type = "response")
  
#   return(evalPredict(testing_set$contains_bug, tmp_lr_pred, testing_set$loc))
# }
# lr_perf <- llply(seq(1, length(project_names), 1), get_lr_perf, .parallel = TRUE)
# lr_perf <- rbindlist(lr_perf, fill=TRUE)

# Performance of Project Aware JIT Model (Logistic Regression)
# print("Running Project aware")

# get_project_aware_lr_perf <- function(i, correct = TRUE) {
#   training_set <- subset(all_projects, all_projects$project != project_names[i])
#   testing_set <- subset(all_projects, all_projects$project == project_names[i])
#   print("created subsets")
#   tmp_project_aware_lr_model <- glmer(contains_bug ~ (norm_entropy | project) +  fix + ns + nd+ nf + age, 
#                                       data = training_set, family = "binomial")
#   print("Trained model")
#   tmp_project_aware_lr_pred <- predict(tmp_project_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "response")
#   tmp_project_aware_lr_pred_corrected <- predict(tmp_project_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "link")
#   tmp_project_aware_lr_pred_corrected <- tmp_project_aware_lr_pred_corrected - coef(summary(tmp_project_aware_lr_model))[1,1] + median(coef(tmp_project_aware_lr_model)[[1]][,2]) + testing_set$norm_entropy * median(coef(tmp_project_aware_lr_model)[[1]][,1])
#   tmp_project_aware_lr_pred_corrected <- inv.logit(tmp_project_aware_lr_pred_corrected)
  
#   if(correct) {
#     return(evalPredict(testing_set$contains_bug, tmp_project_aware_lr_pred_corrected, testing_set$loc))
#   } else {
#     return(evalPredict(testing_set$contains_bug, tmp_project_aware_lr_pred, testing_set$loc))
#   }
# }
# project_aware_lr_perf_correct <- llply(seq(1, length(project_names), 1), get_project_aware_lr_perf, .parallel = TRUE)
# project_aware_lr_perf_correct <- rbindlist(project_aware_lr_perf_correct, fill=TRUE)


# # ===================== RQ4 ========================
# # Model training
# print("RQ4")

# project.aware <- glmer(contains_bug ~ (norm_entropy | project) + fix + ns + nd+ nf + age, 
#                        data = all_projects, family = "binomial")

# # Median Absolute Error for entropy and intercepts
# project.aware.entropy <- coef(project.aware)[[1]][,1]
# project.aware.intercept <- coef(project.aware)[[1]][,2]
# # median(abs(local.entropy - project.aware.entropy)) # MAE for Entropy
# # median(abs(local.intercept - project.aware.intercept)) # MAE for intercepts

# # Goodness-of-fit
# print(r.squaredGLMM(project.aware))

# # Chisq
# Anova(project.aware, Type = 2)

# # Likelihood Ratio Test
# project.aware.noslope <- glmer(contains_bug ~ (1 | project) + fix + ns + nd+ nf + age, 
#                                data = all_projects, family = "binomial")
# mixed.effect.null <- glm(contains_bug ~ fix + ns + nd+ nf + age, 
#                          data = all_projects, family = "binomial")
# anova(project.aware, project.aware.noslope, mixed.effect.null)
