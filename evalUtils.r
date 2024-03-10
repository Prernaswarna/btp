calcIFA <- function(pred, actual) {
  tmp_df <- data.frame(pred = pred, actual = actual)[order(-pred),]
  rownames(tmp_df) <- NULL
  return(min(which(tmp_df$pred > 0.5 & tmp_df$actual == TRUE)))
}

calcPCI20 <- function(pred, loc) {
  tmp_df <- data.frame(pred = pred, loc = loc)[order(-pred),]
  rownames(tmp_df) <- NULL
  tmp_df$cumloc <- cumsum(tmp_df$loc)
  return(max(c(0,which(tmp_df$cumloc <= sum(tmp_df$loc) * 0.2))) / length(pred))
}

calcROC <- function(pred, actual) {
  sortedId <- order(pred, decreasing=TRUE)
  fp <- tp <- fp_prev <- tp_prev <- 0
  nF <- sum(actual == FALSE)
  nT <- sum(actual == TRUE)
  
  if(nF == 0 || nT == 0) {
    return (0)
  }
  
  pred_prev <- -Inf
  ber_min <- Inf
  area <- 0
  rx <- ry <- numeric(length(sortedId))
  n <- 0
  for (i in seq_along(sortedId)) {
    j <- sortedId[i]
    if (pred[j] != pred_prev) {
      area <- area + (fp - fp_prev) * (tp + tp_prev) / 2
      n <- n + 1
      rx[n] <- fp/nF
      ry[n] <- tp/nT
      ber <- (fp/nF + 1 - tp/nT)/2
      if (ber < ber_min) {
        ber_min <- ber
        th <- pred_prev
        rx_best <- fp/nF
        ry_best <- tp/nT
      }
      pred_prev <- pred[j]
      fp_prev <- fp
      tp_prev <- tp
    }
    if (actual[j] == TRUE) {
      tp <- tp + 1
    } else {
      fp <- fp + 1
    }
  }
  area <- area + (fp - fp_prev) * (tp + tp_prev) / 2
  return (area/(nF*nT))
}

evalConfusionMatrix <- function (act, pred, cutoff){
  pred[pred > cutoff] = 1
  pred[pred <= cutoff] = 0
  
  # make the table
  a=table(act, pred)
  
  res=c()
  
  if((ncol(a) == 2) && (nrow(a) == 2)){
    res$acc   = (a[1,1] + a[2,2]) / (a[1,1] +a[1,2]+ a[2,1]+a[2,2])
    res$type1 = a[1,2]/(a[1,2]+a[1,1])
    res$type2 = a[2,1] / (a[2,2] + a[2,1])
    res$precision = a[2,2]/(a[1,2]+a[2,2])
    res$recall = a[2,2]/(a[2,1]+a[2,2])
    res$f  = 2 * res$precision * res$recall / (res$precision + res$recall)
    res$t1 = a[1,2] / (a[1,1] + a[1,2])
    res$t2 = a[2,1] / (a[2,1] + a[2,2])
    res$table = a
  }else{ # exception
    if(ncol(a) == 1){
      # all commits are predicted as non-buggy
      if(sum(pred) == 0){
        res$acc = a[1] / (a[1] + a[2])
        res$type1 = 0
        res$type2 = a[2] / (a[1] + a[2])
        res$precision = 1
        res$recall = 0
        res$f = 0
        res$t1 = 0
        res$t2 = a[2] / (a[2])
        # all commits are predicted as buggy    
      }else{
        res$acc = a[2] / (a[1] + a[2])
        res$type1 = a[1] / (a[1] + a[2])
        res$type2 = 0
        res$precision = a[2] / (a[1] + a[2])
        res$recall = 1
        res$f  = 2 * res$precision * res$recall / (res$precision + res$recall)
        res$t1 = a[1] / (a[1])
        res$t2 = 0
      }
    }else{ # non-buggy commit
      res$acc = 0
      res$type1 = 0
      res$type2 = 0
      res$precision = 0
      res$recall = 0
      res$f = 0
      res$t1 = 0
      res$t2 = 0
    }
  }
  
  res$gscore = 2 * res$recall * (1 - res$type1) / (1 + res$recall - res$type1)
  return (res)
}

# version 0.3

# Normalized Popt [0, 1]
# optmax : maximum auc
# optmin : minimum auc
# auc    : actual auc
calcOPTnorm <- function(optmax,optmin,auc){
  p = (optmax - auc)
  return (1 - (p / (optmax - optmin)))
}

# Popt, which is proposed by Mende et al.
# optauc : maximum auc
# auc    : actual auc
calcOPT <- function(optauc, auc){
  return (1- (optauc - auc))
}

calcOPT20 <- function(optauc, auc){
  return (0.2- (optauc - auc))
}

# cumLOC : x-axis
# cumIndp: y-axis
# norm   : is normalized
calcAUC <- function(cumLOC, cumIndp, norm=TRUE){
  # to considere for diffLOC 0
  mat = cbind(cumLOC, cumIndp)
  tmp = mat[order(mat[,2], decreasing=T),]
  res = tmp[order(tmp[,1], decreasing=F),]  
  cumIndp = res[,2]
  
  # calc diff
  diffLOC <- diff(cumLOC)
  diffIndp <- diff(cumIndp)
  
  # calc AUC of the plot  
  auc <- (diffLOC * cumIndp[2:length(cumIndp)]) - (diffLOC * diffIndp / 2)
  
  # bug fix: + ((cumLOC[1] + cumIndp[1]) / 2) (20091113)
  auc <- sum(auc) + (cumLOC[1] * cumIndp[1] / 2)
  if(norm){
    auc <- auc / (cumLOC[length(cumLOC)] * cumIndp[length(cumIndp)])
  }
  
  return (auc)
}

# cumLOC : x-axis
# cumIndp: y-axis
# norm   : is normalized
calcAUC20 <- function(cumLOC, cumIndp, norm=TRUE){
  # cut at 20%
  square_area = cumLOC[length(cumLOC)] * cumIndp[length(cumIndp)]
  cumLOC <- cumLOC[1:(length(cumLOC) * 0.2)]
  cumIndp <- cumIndp[1:(length(cumIndp) * 0.2)]
  
  # to considere for diffLOC 0
  mat = cbind(cumLOC, cumIndp)
  tmp = mat[order(mat[,2], decreasing=T),]
  res = tmp[order(tmp[,1], decreasing=F),]  
  cumIndp = res[,2]
  
  # calc diff
  diffLOC <- diff(cumLOC)
  diffIndp <- diff(cumIndp)
  
  # calc AUC of the plot  
  auc <- (diffLOC * cumIndp[2:length(cumIndp)]) - (diffLOC * diffIndp / 2)
  
  # bug fix: + ((cumLOC[1] + cumIndp[1]) / 2) (20091113)
  auc <- sum(auc) + (cumLOC[1] * cumIndp[1] / 2)
  if(norm){
    auc <- auc / square_area
  }
  
  return (auc)
}

# indpval: y-axis (e.g., bug density)
# sortval: sort key (e.g., the prediction value)
# loc    : x-axis (e.g., LOC_TOTAL, effort)
# dc     : is decreasing
calcLBC <- function(indpval, sortval, loc, dc=FALSE) {
  # backup row names
  rname = row.names(loc)
  
  # flatting the arguments as list
  sortval <- unlist(sortval)
  loc <- unlist(loc)
  indpval <- unlist(indpval)
  
  # the 1st sort key is sortval but the 2nd sort key is not considered (todo)
  sortedId = order(sortval, decreasing=!dc)
  
  # cumulative summing
  cloc <- cumsum(loc[sortedId])
  cindp <- cumsum(indpval[sortedId])
  
  # calc optimal model
  optId <- order((indpval/(loc+1)), decreasing=TRUE) 
  optcloc <- cumsum(loc[optId])
  optcindp <- cumsum(indpval[optId])
  
  minId  <- order((indpval/(loc+1)), decreasing=FALSE)
  mincloc <- cumsum(loc[minId])
  mincindp <- cumsum(indpval[minId])
  
  # calc AUC of the plot
  auc <- calcAUC(cloc,cindp)
  optauc <- calcAUC(optcloc,optcindp)
  minauc <- calcAUC(mincloc,mincindp)
  
  auc <- as.numeric(auc)
  optauc <- as.numeric(optauc)
  minauc <- as.numeric(minauc)
  
  # calc AUC 20% of the plot
  auc20 <- calcAUC20(cloc,cindp)
  optauc20 <- calcAUC20(optcloc,optcindp)
  minauc20 <- calcAUC20(mincloc,mincindp)
  
  auc20 <- as.numeric(auc20)
  optauc20 <- as.numeric(optauc20)
  minauc20 <- as.numeric(minauc20)
  
  # calc OPT
  opt <- calcOPT(optauc, auc)
  opt_norm <- calcOPTnorm(optauc, minauc, auc)
  
  # calc OPT 20%
  opt20 <- calcOPT20(optauc20, auc20)
  opt20_norm <- calcOPTnorm(optauc20, minauc20, auc20)
  
  # return values
  cumdate = data.frame(LOC=cloc, Indp=cindp)
  row.names(cumdate) = rname
  
  optcumdate = data.frame(LOC=optcloc, Indp=optcindp)
  row.names(optcumdate) = rname
  
  all.bug = sum(indpval[sortedId])
  twenty.bug = sum(indpval[sortedId][cloc < (sum(loc[sortedId]) * 0.2) ])
  
  res = c(cumDate = list(cumdate),
          optcumDate = list(optcumdate),
          OPT = list(opt),
          OPT_Norm = list(opt_norm),
          OPT20 = list(opt20),
          OPT20_Norm = list(opt20_norm),
          AUC = list(auc),
          maxAUC = list(optauc),
          minAUC = list(minauc),
          twenty = list(twenty.bug/all.bug))
  return (res)
}

evalPredict <- function(actual, pred, loc, threshold = 0.5) {
  res <- c()
  
  confMatrix <- evalConfusionMatrix(actual, pred, threshold)
  res$precision <- confMatrix$precision
  res$recall <- confMatrix$recall
  res$f <- confMatrix$f
  res$gscore <- confMatrix$gscore
  
  res$auc <- calcROC(pred, actual)
  res$ifa <- calcIFA(pred, actual)
  res$pci20 <- calcPCI20(pred, loc)
  
  tmp_lbc <- calcLBC(actual, pred, loc)
  res$opt <- tmp_lbc$OPT
  res$opt20 <- tmp_lbc$OPT20
  
  return(res)
}

rsq <- function (x, y) cor(x, y) ^ 2

normalize3 <- function(training_set, scale_metrics, all_projects) {
  print("inside normalize function")
#  vec <- training_set[,"norm_entropy"]
#     for(j in 1:3) {
#       print(vec[j])
#     }
  for (i in 1:length(scale_metrics)) {
    # print(scale_metrics[i])
    # print(length(projects[[training_project]][,scale_metrics[i]]))
    mean_src <- mean(training_set[,scale_metrics[i]])
    sd_src <- sd(training_set[,scale_metrics[i]])
    # print(mean_src)
    # print(sd_src)
    # print(length(all_projects[,scale_metrics[i]]))
    # vec <- all_projects[,scale_metrics[i]]
    # for(j in 1:3) {
    #   print(vec[j])
    # }
    all_projects[,scale_metrics[i]] <- (all_projects[,scale_metrics[i]] - mean_src)/sd_src
    # vec <- all_projects[,scale_metrics[i]]
    # for(j in 1:3) {
    #   print(vec[j])
    # }
  }

  # training_set <- subset(all_projects, all_projects$project != project_names[1])
  # vec <- training_set[,"norm_entropy"]
  #   for(j in 1:3) {
  #     print(vec[j])
  #   }
  print("Done")
  
  return(all_projects)
}


get_prob <- function(data_p1) {
data_list <- list(data_p1)
# head(data_p1)
data_list <- lapply(data_list,"*",1000)
data_p1 <- data_list[[1]]
data_p1 <- ceiling(data_p1)
# head(data_p1)

# mini1 <- min(data_p1)
# maxi1 <- max(data_p1)

# print("minimum of project")
# print(mini1)
# print("maximum of project")
# print(maxi1)

prob_p1 <- prop.table(table(data_p1))
# head(prob_p1)
# print("value")
prob_p1 <- as.matrix(prob_p1)
# head(prob_p1)
return(prob_p1)
}

get_mini <- function(prob_p1) {
  existing_values <- rownames(prob_p1)
  existing_values <- strtoi(existing_values)
  # print(existing_values[1])
  return(existing_values[1])
}


get_maxi <- function(prob_p1) {
  existing_values <- rownames(prob_p1)
  existing_values <- strtoi(existing_values)
  # print(existing_values[length(existing_values)])
  return(existing_values[length(existing_values)])
}

prob_distr <- function(mini1, mini2, maxi1, maxi2, prob_p1) {  
# print("project probabilities")
existing_values <- rownames(prob_p1)
existing_values <- strtoi(existing_values)
# head(existing_values)

all_values <- c()
j= min(mini1,mini2)
ind=1
curr=existing_values[ind]
while(j<maxi1) {
  while(j<curr) {
    all_values <- append(all_values, 0)
    j<-j+1
  }
  if(j==curr) {
    all_values <- append(all_values, prob_p1[ind])
    j<-j+1
    ind <- ind+1
    curr = existing_values[ind]
  }
}
all_values <- append(all_values, prob_p1[ind])
j<-j+1
while(j<=max(maxi1,maxi2)) {
  all_values <- append(all_values, 0)
  j<-j+1
} 
# print(length(all_values))
return(all_values)
}

kld <- function(p, q, base = c("log", "log2", "log10"), margin = FALSE) {
  base <- match.arg(base)
  
  klds <- p * do.call(base, list(p / q))
  if (margin) {
    return(sum(klds, na.rm = TRUE))
  } else {
    klds
  }
}

jsd <- function(p, q, base = c("log", "log2", "log10"), margin = FALSE) {
  base <- match.arg(base)
  
  m <- (p + q) / 2
  
  jsds <- 0.5 * (kld(p , m, base) + kld(q , m, base))
  
  if (margin) {
    return(sum(jsds, na.rm = TRUE))
  } else {
    jsds
  }
}
