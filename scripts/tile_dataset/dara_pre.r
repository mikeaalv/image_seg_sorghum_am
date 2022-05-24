# prepare the images data into one folder with table of locaiton information
rm(list=ls())
options(warn=1)
options(stringsAsFactors=FALSE)
options(digits=15)
require(stringr)
require(magrittr)
require(fs)
require(plyr)
paredir="/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/am_fungi_segmentation/data/AM_classify2/"
tardir="/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/am_fungi_segmentation/results/tiles/data/combined/"
setwd(tardir)
matchpat="*.jpg"
sets=c("train","validate","test")
for(set_ele in sets){
  listfiles=list.files(path=paste0(paredir,set_ele),pattern=matchpat,full.names=FALSE,recursive=TRUE)
  infortabe=data.frame(from=paste0(paredir,set_ele,"/",listfiles),to=paste0(tardir,set_ele,"/",listfiles))
  file_copy(infortabe[,1],infortabe[,2],overwrite=TRUE)
  tab_exist=read.table(paste0(tardir,set_ele,"/regiondata.csv"),sep="\t",header=TRUE)
  tab_add=read.table(paste0(paredir,set_ele,"/regiondata.csv"),sep="\t",header=TRUE)
  tab_comb=rbind.fill(tab_exist,tab_add)
  write.table(tab_comb,file=paste0(set_ele,"/regiondata.csv"),sep="\t",row.names=FALSE,qmethod="double")
}
