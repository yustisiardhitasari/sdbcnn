# Stack multiple bands

setwd("D:/sdb/img")
f.aoi <- "aoi1"

library(raster)
library(shapefiles)
library(rgdal)

# SENTINEL-2
b2 <- raster("T19QGV_20190108T150721_B02_20m.jp2")
b3 <- raster("T19QGV_20190108T150721_B03_20m.jp2")
b4 <- raster("T19QGV_20190108T150721_B04_20m.jp2")
b8 <- raster("T19QGV_20190108T150721_B08_20m.jp2")
b11 <- raster("T19QGV_20190108T150721_B11_20m.jp2")
b12 <- raster("T19QGV_20190108T150721_B12_20m.jp2")

# RGBNSS
img.stack <- stack(b4, b3, b2, b8, b11, b12)
names(img.stack) <- c('red', 'green', 'blue', 'nir', 'swir1', 'swir2')
writeRaster(img.stack, "ponceRGBNSS_S2_190108.tif")

Clip raster by extent
shp.aoi <- shapefile(paste("../shp/ponce_", f.aoi, ".shp", sep=""))
img.clip <- crop(img.stack, extent(shp.aoi))
names(img.clip) <- c('red', 'green', 'blue', 'nir', 'swir1', 'swir2')
writeRaster(img.clip, paste("../data/ponceRGBNSS_S2_190108", "_", f.aoi, ".tif", sep=""))
