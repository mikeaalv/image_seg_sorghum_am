rm(list=ls())
options(warn=1)
options(stringsAsFactors=FALSE)
options(digits=15)
require(stringr)
require(magrittr)
require(ggplot2)
require(dplyr)
require(reshape2)
resdir="/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/am_fungi_segmentation/results/run_am_seg_bigdataset/res/"
setwd(resdir)
infortab=read.table(file="submitlist.tab",sep="\t",header=TRUE)
seqdir=1:nrow(infortab)
perftab=c()
segtypes=c("bbox","segm")
evaltypes=c("AP","AP50")
datasettypes=c("validate","test")
for(seqi in seqdir){
  record=infortab[seqi,]
  fold=paste0("./",seqi,"/")
  output=list.files(fold,"testmodel.*out")
  outputlines=readLines(paste0(fold,output))
  runname=str_replace_all(record[,"names"],pattern="_.*",replacement="")
  randseed=record[,"random_seed"]
  # block index
  set_ind=list(validate=max(str_which(outputlines,pattern=fixed("am_validate"))),
               test=max(str_which(outputlines,pattern=fixed("am_test"))))
  infor_line_ind=str_which(outputlines,pattern="OrderedDict")
  valvec=c()
  segvec=c()
  evalvec=c()
  setvec=c()
  for(datasettype in datasettypes){
    if(datasettype=="validate"){
      lineshere=outputlines[infor_line_ind[infor_line_ind>set_ind[["validate"]]&infor_line_ind<set_ind[["test"]]]]
    }else if(datasettype=="test"){
      lineshere=outputlines[infor_line_ind[infor_line_ind>set_ind[["test"]]]]
    }
    # double split the string into 2-level list
    split1=str_split(lineshere,"\\),\\s\\(")
    splitlevel2<-function(x,arg1) {str_split(x,",")}
    split2<-lapply(split1,splitlevel2)[[1]]
    #
    for(segtypei in seq(length(segtypes))){
      strarray=split2[[segtypei]]
      for(evaltypei in seq(length(evaltypes))){
        locind=str_which(strarray,pattern=fixed(paste0("'",evaltypes[evaltypei],"'")))
        valvec=c(valvec,as.numeric(str_extract(strarray[locind],patter="[\\d\\.]+$")))
        segvec=c(segvec,segtypes[segtypei])
        evalvec=c(evalvec,evaltypes[evaltypei])
        setvec=c(setvec,datasettype)
      }
    }
  }
  perftab=rbind(perftab,
                data.frame(run=rep(runname,times=length(evalvec)),seed=rep(randseed,times=length(evalvec)),evaluation=evalvec,set=setvec,segmentation=segvec,value=valvec))
}
# test+segm+mean
perftab %>% filter(set=="test"&segmentation=="segm") %>% group_by(run,evaluation) %>% summarise(mean=mean(value)) %>% dcast(run~evaluation,value.var='mean')
# test+bbox+mean
perftab %>% filter(set=="test"&segmentation=="bbox") %>% group_by(run,evaluation) %>% summarise(mean=mean(value)) %>% dcast(run~evaluation,value.var='mean')
# test+segm+mean+std
perftab %>% filter(set=="test"&segmentation=="segm") %>% group_by(run,evaluation) %>% summarise(disp=mean(value),sd=sd(value))
# plotting
