# prepare the images data into one folder with table of locaiton information
rm(list=ls())
options(warn=1)
options(stringsAsFactors=FALSE)
options(digits=15)
require(stringr)
require(magrittr)
require(fs)
paredir="/work/aelab/AMF/AMF Imaging/0_Image_Collection/ZEISS Primo Star/Georgia"
tardir="/scratch/yw44924/amf_inference/data/test"
setwd(tardir)
matchpat="*.jpg"
listfiles=list.files(path=paredir,pattern=matchpat,full.names=TRUE,recursive=TRUE)
seq(length(listfiles)) %>% paste0(.,".jpg") -> newfiles
infortabe=data.frame(location=listfiles,filenames=newfiles)
file_copy(listfiles,newfiles,overwrite=TRUE)
write.table(infortabe,file="locatab.txt",sep=",",row.names=FALSE)
