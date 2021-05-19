# Satellite image correction using mean deep water area (DWA)

# Setting
setwd("D:/sdb/data")
f.path <- "ponceRGBNSS_S2_190108"
f.aoi <- "aoi1"
f.dwa <- "dwa1"

# Import packages
library(raster)
library(rgdal)

# Read data
img.rgb <- stack(readGDAL(paste("../img/", f.path, "_", f.aoi, ".tif", sep="")))
img.dwa <- stack(readGDAL(paste("../img/", f.path, "_", f.dwa, ".tif", sep="")))

# Mean DWA
mean.DWA <- cellStats(x=img.dwa, stat='mean')
img.x.mean <- log(img.rgb - mean.DWA)
names(img.x.mean) <- c("red", "green", "blue", "nir", "swir1", "swir2")
writeRaster(img.x.mean, paste(f.path, "_", f.aoi, "_", f.dwa, "_crctm.tif", sep=""))
