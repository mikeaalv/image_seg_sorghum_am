# prepare the images data into one folder with table of locaiton information
rm(list=ls())
options(warn=1)
options(stringsAsFactors=FALSE)
options(digits=15)
require(stringr)
require(magrittr)
require(fs)
paredir="/work/aelab/AMF/AMF Imaging/0_Image_Collection/ZEISS Primo Star/Georgia/2021/Experiment002_Greenhouse_14_Accession/1_JPEG/"
tardir="/scratch/yw44924/amf_inference3/data/test"
setwd(tardir)
matchpat="*.jpg"
except='Additional images and duplicates'
listfiles=list.files(path=paredir,pattern=matchpat,full.names=TRUE,recursive=TRUE)
ind_expt=str_which(string=listfiles,pattern=fixed(except))
listfiles=listfiles[-ind_expt]
seq(length(listfiles)) %>% paste0(.,".jpg") -> newfiles
infortabe=data.frame(location=listfiles,filenames=newfiles)
file_copy(listfiles,newfiles,overwrite=TRUE)
write.table(infortabe,file="locatab.txt",sep=",",row.names=FALSE)
