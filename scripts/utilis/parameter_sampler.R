rm(list=ls())
options(warn=1)
options(stringsAsFactors=FALSE)
options(digits=15)
require(stringr)
require(magrittr)
require(R.matlab)
require(ggplot2)
require(xml2)
require(scales)
require(foreach)
require(doMC)
require(readr)
require(abind)
dir=""
shelltempt=c(paste0(dir,"am_seg_train.sh"))
infortab=read.table(file="submitlist.tab",sep="\t",header=TRUE)
# infortab[which(is.na(infortab[,"addon"])),"addon"]=""
# seqdir=1:nrow(infortab)
seqdir=1:21
colnam=colnames(infortab)
for(irow in seqdir){
  infor=infortab[irow,]
  system(paste0("mkdir ",irow))
  setwd(paste0("./",irow))
  # system(paste0("cp ../*.R ."))
  system(paste0("cp ../*.sh ."))
  system(paste0("cp ../*.py ."))
  shellscript=shelltempt[1]
  normalize_flag_str=""
  batchnorm_flag_str=""
  if("addon"%in%colnam && str_detect(string=infor[,"addon"],pattern="No_input_normalization")){
    normalize_flag_str=paste0("--normalize-flag N")
  }
  if("addon"%in%colnam && str_detect(string=infor[,"addon"],pattern="No_batch_normliaztion")){
    batchnorm_flag_str=paste0("--batchnorm-flag N")
  }
  dropout_rate=0.0
  if("addon"%in%colnam && str_detect(string=infor[,"addon"],pattern="^dp")){
    dropout_rate=str_replace_all(string=infor[,"addon"],pattern="^dp",replacement="")
  }
  if("addon"%in%colnam && str_detect(string=infor[,"addon"],pattern="^scheduler")){
    scheduler=str_replace_all(string=infor[,"addon"],pattern="^scheduler\\_",replacement="")
  }
  rnnadd=""
  if(str_detect(string=infor[,"net_struct"],pattern="\\_rnn")){
    rnnadd="--rnn-struct 1"
  }
  freeze_tillstr=""
  if(infor[,"freeze"]!=""){
    freeze_tillstr=paste0("--freeze-at"," ",infor[,"freeze"])
  }
  lines=readLines(shellscript)
  chline_ind=str_which(string=lines,pattern="^time")
  lineend=str_extract_all(string=lines[chline_ind],pattern="\\w+>>.*$")[[1]]
  lines[chline_ind]=paste(paste0("time python am_seg_train.py "),
              "--epochs",infor[,"epochs"],
              "--learning-rate",infor[,"learning_rate"],
              "--seed",infor[,"random_seed"],
              "--net-struct",infor[,"net_struct"],
              "--aug-flag",infor[,"aug_flag"],
              normalize_flag_str,batchnorm_flag_str,
              freeze_tillstr,
              rnnadd,
              lineend,
              sep=" "
            )
  newfile=paste0(str_replace(string=shellscript,pattern="\\.sh",replacement=""),infor[1],".sh")
  cat(lines,file=newfile,sep="\n")
  submitcommand=paste0("sbatch ",newfile)
  print(submitcommand)
  system(submitcommand)
  setwd("../")
}
