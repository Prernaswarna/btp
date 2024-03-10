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
library(FSelector)


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
                                             fix = col_logical(),commit_hash = col_skip(), commit_message = col_skip(), fileschanged = col_skip(), 
                                             fixes = col_skip(),ndev = col_skip(),))
  
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
all_projects$project <- as.factor(all_projects$project)


head(all_projects)


# # for(i in 1:length(project_names)) {
#     print(i)
#     tmp_table <- table(all_projects$contains_bug, all_projects$author_name)
#     print("author_name")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$author_date_unix_timestamp)
#     print("author_date_unix_timestamp")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$author_email)
#     print("author_email")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$author_date)
#     print("author_date")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$fix)
#     print("fix")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$classification)
#     print("classification")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$linked)
#     print("linked")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$ns)
#     print("ns")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$nf)
#     print("nf")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$lt)
#     print("lt")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$ndev)
#     print("ndev")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$age)
#     print("age")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$nuc)
#     print("nuc")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$exp)
#     print("exp")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$rexp)
#     print("rexp")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$sexp)
#     print("sexp")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$glm_probability)
#     print("glm_probability")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$loc)
#     print("loc")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$norm_entropy)
#     print("norm_entropy")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$language)
#     print("language")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$nlanguage)
#     print("nlanguage")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$audience)
#     print("audience")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$ui)
#     print("ui")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$database)
#     print("database")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$LOC)
#     print("LOC")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$nfile)
#     print("nfile")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$ncommit)
#     print("ncommit")
#     print(chisq.test(tmp_table))
#     tmp_table <- table(all_projects$contains_bug, all_projects$relative_churn)
#     print("relative_churn")
#     print(chisq.test(tmp_table))
# # }

# for(i in 1:length(project_names)) {
#     print(i)
#     print("author_name")
#     print(gain.ratio(contains_bug~., all_projects))
# }


# print(cor(all_projects$contains_bug, all_projects$author_date_unix_timestamp, method="pearson"))


# for(i in 1:length(project_names)) {
    # print(i)
# transform(projects[[i]], author_name=as.numeric(as.factor(projects[[i]]$author_name)))
# print(projects[[i]]$author_name)
    # print("author_name")
    # val=cor(projects[[i]]$contains_bug, projects[[i]]$author_name, method="spearman")
    # # print(val)
    print("author_date_unix_timestamp")
    print(cor(all_projects$contains_bug, all_projects$author_date_unix_timestamp, method="pearson"))
    # print("author_email")
    # print(cor(projects[[i]]$contains_bug, projects[[i]]$author_email, method="spearman"))
    # print("author_date")
    # print(cor(projects[[i]]$contains_bug, projects[[i]]$author_date, method="spearman"))
    print("fix")
    print(cor(all_projects$contains_bug, all_projects$fix, method="pearson"))
    # print("classification")
    # print(cor(projects[[i]]$contains_bug, projects[[i]]$classification, method="spearman"))
    print("linked")
    print(cor(all_projects$contains_bug, all_projects$linked, method="pearson"))
    print("ns")
    print(cor(all_projects$contains_bug, all_projects$ns, method="pearson"))
    print("nf")
    print(cor(all_projects$contains_bug, all_projects$nf, method="pearson"))
    print("lt")
    print(cor(all_projects$contains_bug, all_projects$lt, method="pearson"))
    print("ndev")
    print(cor(all_projects$contains_bug, all_projects$ndev, method="pearson"))
    print("age")
    print(cor(all_projects$contains_bug, all_projects$age, method="pearson"))
    print("nuc")
    print(cor(all_projects$contains_bug, all_projects$nuc, method="pearson"))
    print("exp")
    print(cor(all_projects$contains_bug, all_projects$exp, method="pearson"))
    print("rexp")
    print(cor(all_projects$contains_bug, all_projects$rexp, method="pearson"))
    print("sexp")
    print(cor(all_projects$contains_bug, all_projects$sexp, method="pearson"))
    print("glm_probability")
    print(cor(all_projects$contains_bug, all_projects$glm_probability, method="pearson"))
    print("loc")
    print(cor(all_projects$contains_bug, all_projects$loc, method="pearson"))
    print("norm_entropy")
    print(cor(all_projects$contains_bug, all_projects$norm_entropy, method="pearson"))
    # print("language")
    # print(cor(projects[[i]]$contains_bug, projects[[i]]$language, method="spearman"))
    print("nlanguage")
    print(cor(all_projects$contains_bug, all_projects$nlanguage, method="pearson"))
    # print("audience")
    # print(cor(projects[[i]]$contains_bug, projects[[i]]$audience, method="spearman"))
    # print("ui")
    # print(cor(projects[[i]]$contains_bug, projects[[i]]$ui, method="spearman"))
    print("database")
    print(cor(all_projects$contains_bug, all_projects$database, method="pearson"))
    print("LOC")
    print(cor(all_projects$contains_bug, all_projects$LOC, method="pearson"))
    print("nfile")
    print(cor(all_projects$contains_bug, all_projects$nfile, method="pearson"))
    print("ncommit")
    print(cor(all_projects$contains_bug, all_projects$ncommit, method="pearson"))
    print("relative_churn")
    print(cor(all_projects$contains_bug, all_projects$relative_churn, method="pearson"))
# }