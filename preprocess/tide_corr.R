# Depth correction based on tide at local time

# Setting
setwd("D:/sdb/depthref")
f.path <- "ponce_depth_aoi1_10m"

# Import packages
library(raster)
library(rgdal)

# Read data
img.depth <- stack(readGDAL(paste(f.path, ".tif", sep="")))

# Tide correction
tide <- 0.16
img.x.crct <- img.depth - tide # minus because depth values are negative numbers. Use plus if the opposite
writeRaster(img.x.crct, paste(f.path, "_crct.tif", sep=""))
