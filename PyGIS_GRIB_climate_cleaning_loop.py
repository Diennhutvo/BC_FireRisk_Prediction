import processing
import sys
import os
import random
from datetime import datetime, timezone, timedelta
from qgis.analysis import QgsNativeAlgorithms
from qgis.PyQt.QtCore import QVariant
from pyproj import Transformer
import re
import math
from osgeo import gdal
from processing.core.Processing import Processing
from qgis.core import (QgsVectorLayer, QgsProject, QgsProcessingContext, 
                       QgsProcessingFeedback, edit, QgsApplication, 
                       QgsProcessingFeatureSourceDefinition, QgsRasterLayer,
                       QgsRasterBandStats, QgsField, QgsFeature, QgsGeometry,
                       QgsPointXY, QgsVectorFileWriter, 
                       QgsCoordinateReferenceSystem, QgsProcessingProvider
)
from processing.algs.gdal.GdalAlgorithmProvider import GdalAlgorithmProvider
import csv


# Initialize Processing framework
Processing.initialize()

# Register built-in QGIS and GDAL providers
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
QgsApplication.processingRegistry().addProvider(GdalAlgorithmProvider())

print("✅ GDAL Processing tools enabled.")

# --- Load Canada provinces shapefile
base_dir = 'C:/Users/tdoa2/Downloads/Spatial data cleaning/'
canada_provinces_path = os.path.join(base_dir, 'Map_of_Canada/lpr_000b16a_e.shp')
canada_layer = QgsVectorLayer(canada_provinces_path, 'Canada Provinces', 'ogr')

if not canada_layer.isValid():
    print("❌ Canada provinces layer failed to load.")
else:
    print("✅ Canada provinces layer created.")

# --- Extract British Columbia boundary
bc_output_path = os.path.join(base_dir, 'Map_of_Canada/BC_boundary.shp')

processing.run("native:extractbyattribute", {
    'INPUT': canada_layer,
    'FIELD': 'PREABBR', 
    'OPERATOR': 0,  # '='
    'VALUE': 'B.C.', 
    'OUTPUT': bc_output_path
})
print("✅ BC boundary extracted and saved to:", bc_output_path)

# --- Load BC layer
bc_boundary_layer = QgsVectorLayer(bc_output_path, 'BC Boundary', 'ogr')
if bc_boundary_layer.isValid():
    print("✅ BC Boundary layer created.")
else:
    print("❌ Failed to load BC Boundary layer.")

# === STEP: Reproject BC Boundary to EPSG:3347 ===
reprojected_bc_boundary = os.path.join(base_dir, 'Map_of_Canada/BC_boundary_epsg3347.shp')

processing.run("native:reprojectlayer", {
    'INPUT': bc_output_path,
    'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:3347'),
    'OUTPUT': reprojected_bc_boundary
})
print("✅ Reprojected BC boundary to EPSG:3347")

# Load reprojected BC boundary
bc_boundary_layer_3347 = QgsVectorLayer(reprojected_bc_boundary, 'BC Boundary (EPSG:3347)', 'ogr')
if bc_boundary_layer_3347.isValid():
    QgsProject.instance().addMapLayer(bc_boundary_layer_3347)
    print("✅ Reprojected BC boundary layer created successfully.")
else:
    print("❌ Failed to load reprojected BC boundary layer.")

# --- Clip and reproject Fuel Raster
fuel_raster_path = os.path.join(base_dir, 'National_FBP_Fueltypes_version2014b/nat_fbpfuels_2014b.tif')
clipped_raster_temp = os.path.join(base_dir, 'National_FBP_Fueltypes_version2014b/temp_clipped_fuel.tif')
final_clipped_raster = os.path.join(base_dir, 'National_FBP_Fueltypes_version2014b/BC_fuel_type_epsg3347.tif')

# Clip raster to BC
processing.run("gdal:cliprasterbymasklayer", {
    'INPUT': fuel_raster_path,
    'MASK': reprojected_bc_boundary,
    'SOURCE_CRS': None,
    'TARGET_CRS': None,
    'NODATA': -9999,
    'ALPHA_BAND': False,
    'CROP_TO_CUTLINE': True,
    'KEEP_RESOLUTION': True,
    'OPTIONS': '',
    'DATA_TYPE': 0,
    'EXTRA': '',
    'OUTPUT': clipped_raster_temp
})
print("✅ Temporary clipped fuel raster created.")

# Check and reproject if the clipped raster exists
if os.path.exists(clipped_raster_temp):
    processing.run("gdal:warpreproject", {
        'INPUT': clipped_raster_temp,
        'SOURCE_CRS': None,
        'TARGET_CRS': 'EPSG:3347',
        'RESAMPLING': 0,  # Nearest neighbor
        'NODATA': None,
        'TARGET_RESOLUTION': None,
        'OPTIONS': '',
        'DATA_TYPE': 0,
        'TARGET_EXTENT': None,
        'TARGET_EXTENT_CRS': None,
        'MULTITHREADING': False,
        'OUTPUT': final_clipped_raster
    })
    print("✅ Reprojected raster to EPSG:3347 successfully.")

    reprojected_layer = QgsRasterLayer(final_clipped_raster, "BC Fuel Type (EPSG:3347)")
    if reprojected_layer.isValid():
        print("✅ Reprojected raster added to project.")
        QgsProject.instance().addMapLayer(reprojected_layer)
    else:
        print("❌ Failed to load reprojected raster.")
else:
    print(f"❌ File not found: {clipped_raster_temp}")


# === PARAMETERS
start_year = 2000
end_year = 2001

# --- Months dictionary
month_words = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


# -- Total fires
fire_counts = {}  # Store fire point count for each (year, month)
non_fire_counts = {} # Store planned non-fire point count

# == FUNCTIONS
# -- Get fire hotspots files
def get_hotspots(year):
    # Load Hotspot shapefile
    hotspot_path = os.path.join(base_dir, f"Point_data/Hotspot data/{year}_hotspots/{year}_hotspots.shp")
    hotspot_layer = QgsVectorLayer(hotspot_path, f"Hotspots {year}", 'ogr')
    
    if not hotspot_layer.isValid():
        print("❌ Hotspot shapefile failed to load.")
        return
    else:
        print(f"✅ {year} Hotspot layer created.")
    return hotspot_layer

# Get fire data by month
def get_monthly_hotspot_data(month_num, month_name, year, hotspot_layer):
    # Detect date field
    field_names = [field.name() for field in hotspot_layer.fields()]
    if 'REP_DATE' in field_names:
        date_field = 'REP_DATE'
    elif 'rep_date' in field_names:
        date_field = 'rep_date'
    else:
        print(f"⚠️ {year} hotspot layer: No REP_DATE or rep_date field.")
        return None

    # Try extracting a sample date value
    sample_date = None
    for feat in hotspot_layer.getFeatures():
        date_value = feat[date_field]
        if date_value:
            sample_date = str(date_value)
            break

    if not sample_date:
        print(f"⚠️ {year} hotspot layer: No valid date value found.")
        return None

    # Detect date separator
    if "-" in sample_date:
        separator = "-"
    elif "/" in sample_date:
        separator = "/"
    else:
        print(f"⚠️ {year} hotspot layer: Unknown date format '{sample_date}'.")
        return None

    print(f"📅 {year} hotspot: Detected date separator '{separator}'.")

    # Build and run the expression to filter by month and year
    expression = f"month(\"{date_field}\") = {month_num} AND year(\"{date_field}\") = {year}"
    result = processing.run("native:extractbyexpression", {
        'INPUT': hotspot_layer,
        'EXPRESSION': expression,
        'OUTPUT': 'memory:'
    })

    filtered_layer = result['OUTPUT']
    if not filtered_layer.isValid() or filtered_layer.featureCount() == 0:
        print(f"⚠️ {month_name} {year}: No hotspot features found after filtering.")
        return None

    filtered_layer.setName(f"Filtered_{year}_{month_name}")
    print(f"✅ {month_name} {year}: Filtered layer with {filtered_layer.featureCount()} features.")

    # Clip to BC
    result = processing.run("native:clip", {
        'INPUT': filtered_layer,
        'OVERLAY': bc_output_path,
        'OUTPUT': 'memory:'
    })

    clipped_layer = result['OUTPUT']
    if not clipped_layer.isValid() or clipped_layer.featureCount() == 0:
        print(f"❌ {month_name} {year}: Clipping to BC failed or returned no features.")
        return None

    clipped_layer.setName(f"BC hotspots {month_name} {year}")
    print(f"✅ {month_name} {year}: Clipped to BC with {clipped_layer.featureCount()} features.")

    # Reproject to EPSG:3347
    result = processing.run("native:reprojectlayer", {
        'INPUT': clipped_layer,
        'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:3347'),
        'OUTPUT': 'memory:'
    })

    hotspot_layer_3347 = result['OUTPUT']
    if not hotspot_layer_3347.isValid() or hotspot_layer_3347.featureCount() == 0:
        print(f"❌ {month_name} {year}: Reprojection failed.")
        return None

    hotspot_layer_3347.setName(f"BC hotspots {month_name} {year} EPSG:3347")
    print(f"✅ {month_name} {year}: Reprojected to EPSG:3347 with {hotspot_layer_3347.featureCount()} features.")

    return hotspot_layer_3347
    



# -- Get climate data
def climate_extraction(year, month_name, month_num):
    yearly_climate =os.path.join(base_dir, f"climate_data/GRIB_climate_data/{year}/{month_name}")
    os.makedirs(yearly_climate, exist_ok=True)
    
    if year % 2 == 0:
        climate_grib_path = os.path.join(base_dir, f"climate_data/GRIB_climate_data/{year}-{year + 1}.grib")
        bc_climate_temp = os.path.join(yearly_climate, f"BC_temp_{year}-{year + 1}_{month_name}.tif")
        bc_climate_final = os.path.join(yearly_climate, f"BC_{year}-{year + 1}_{month_name}.tif")
    else:
        climate_grib_path = os.path.join(base_dir, f"climate_data/GRIB_climate_data/{year - 1}-{year}.grib")
        bc_climate_temp = os.path.join(yearly_climate, f"BC_temp_{year - 1}-{year}_{month_name}.tif")
        bc_climate_final = os.path.join(yearly_climate, f"BC_{year - 1}-{year}_{month_name}.tif")

    ds = gdal.Open(climate_grib_path)

    if ds is None:
        print("❌ Could not open GRIB file.")
        return

    band_count = ds.RasterCount
    print(f"📊 Total bands: {band_count}")

    target_year = year
    target_month = month_num
    selected_band_indices = []

    for i in range(1, band_count + 1):
        band = ds.GetRasterBand(i)
        metadata = band.GetMetadata()

        valid = metadata.get("GRIB_VALID_TIME")
        comment = metadata.get("GRIB_COMMENT", "").lower()

        if valid:
            valid_dt = datetime.fromtimestamp(int(valid), tz=timezone.utc)
            adjusted_dt = valid_dt

            if any(keyword in comment for keyword in ["[m]", "precipitation", "total"]):
                adjusted_dt += timedelta(days=1)

            if adjusted_dt.year == target_year and adjusted_dt.month == target_month:
                selected_band_indices.append(i)
                print(f"🟢 Band {i} → {adjusted_dt.strftime('%Y-%m-%d')} (adjusted from {valid_dt.strftime('%Y-%m-%d')})")
            else:
                print(f"⚪️ Band {i} skipped → {adjusted_dt.strftime('%Y-%m-%d')}")
        else:
            print(f"⚠️ Band {i} has no VALID_TIME")

    if not selected_band_indices:
        print(f"⚠️ No bands selected for {month_name} {year}. Skipping...")
        return

    print(f"\n📦 Bands selected for {target_year}-{target_month:02d}: {selected_band_indices}")

    temp_band_files = []

    # Step 1: Save to in-memory bands
    for i, band_index in enumerate(selected_band_indices):
        temp_path = f"/vsimem/temp_band_{i+1}.tif"
        gdal.Translate(temp_path, climate_grib_path, bandList=[band_index])
        temp_band_files.append(temp_path)

    # Step 2: In-memory VRT
    output_vrt = f"/vsimem/{month_name}_{year}.vrt"
    gdal.BuildVRT(output_vrt, temp_band_files, separate=True)

    # Step 3: Write final TIF to disk
    output_tif = os.path.join(yearly_climate, f"{month_name}_{year}.tif")
    gdal.Translate(output_tif, output_vrt)

    # Step 4: Clip raster
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT': output_tif,
        'MASK': bc_boundary_layer_3347,
        'SOURCE_CRS': None,
        'TARGET_CRS': None,
        'NODATA': -9999,
        'ALPHA_BAND': False,
        'CROP_TO_CUTLINE': True,
        'KEEP_RESOLUTION': True,
        'OPTIONS': '',
        'DATA_TYPE': 0,
        'EXTRA': '',
        'OUTPUT': bc_climate_temp
    })

    # Step 5: Reproject raster
    if os.path.exists(bc_climate_temp):
        processing.run("gdal:warpreproject", {
            'INPUT': bc_climate_temp,
            'SOURCE_CRS': None,
            'TARGET_CRS': 'EPSG:3347',
            'RESAMPLING': 0,
            'NODATA': None,
            'TARGET_RESOLUTION': None,
            'OPTIONS': '',
            'DATA_TYPE': 0,
            'TARGET_EXTENT': None,
            'TARGET_EXTENT_CRS': None,
            'MULTITHREADING': False,
            'OUTPUT': bc_climate_final
        })
        print("✅ Reprojected raster to EPSG:3347 successfully.")

        bc_climate_raster_epsg3347 = QgsRasterLayer(bc_climate_final, f"BC Climate raster {month_name} {year} (EPSG:3347)")
    else:
        print("❌ Failed to reproject raster to EPSG:3347.")
        return

    # Step 6: Load into QGIS
    if bc_climate_raster_epsg3347.isValid():
        print("✅ Loaded reprojected climate raster successfully.")
    else:
        print("❌ Failed to load reprojected raster.") 
    
    # === Input paths ===
    output_folder = os.path.join(yearly_climate, f"Filled_Bands_{month_name}_{year}")
    bc_climate_filled = os.path.join(output_folder, 'Filled_Stacked_Climate.tif')

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    
    # === Open raster and check band count ===
    ds_final = gdal.Open(bc_climate_final)
    if ds_final is None:
        print(f"❌ Failed to open reprojected raster {bc_climate_final}")
        return
    
    band_count = ds_final.RasterCount
    print(f"📊 Found {band_count} bands to fill.")
    
    filled_band_paths = []
    
    # === Loop through each band and fill nodata ===
    for i in range(1, band_count + 1):
        output_band = os.path.join(output_folder, f'filled_band_{i}.tif')
        print(f"🌀 Filling NoData for Band {i}...")
    
        processing.run("gdal:fillnodata", {
            'INPUT': bc_climate_final,
            'BAND': i,
            'MASK_LAYER': None,
            'DISTANCE': 10,
            'ITERATIONS': 0,
            'NO_MASK': False,
            'OUTPUT': output_band
        })
    
        if os.path.exists(output_band):
            print(f"✅ Band {i} filled and saved to {output_band}")
            filled_band_paths.append(output_band)
        else:
            print(f"❌ Failed to process Band {i}")
    
    # === Stack all filled bands into one multi-band raster ===
    print("🧱 Stacking filled bands into one raster...")
    
    vrt_path = os.path.join(output_folder, 'temp_stack.vrt')
    gdal.BuildVRT(vrt_path, filled_band_paths, separate=True)
    gdal.Translate(bc_climate_filled, vrt_path)
    
    print(f"🎉 All bands filled and stacked! Final output: {bc_climate_filled}")
    
    layer_name = f"Filled BC Climate {month_name} {year}"
    bc_climate_layer = QgsRasterLayer(bc_climate_filled, layer_name)
    if bc_climate_layer.isValid():
        print(f"✅ Reprojected climate raster '{layer_name}' created successfully.")
    else:
        print("❌ Failed to load reprojected raster.")
    
    # === Clean up in-memory VRT and temp band files ===
    gdal.Unlink(output_vrt)
    for path in temp_band_files:
        gdal.Unlink(path)
    
    return bc_climate_filled


# -- Generate random non-fire points
def gen_non_fire_points(year, month_name, non_fire_count):
    """
    Generates random non-fire points within the BC boundary.

    Parameters:
    - year (int): The year for which points are being generated.
    - month_name (str): Month name (e.g., 'January').
    - non_fire_count (int): Number of non-fire points to generate (either equal to fire count or 400).
    
    Returns:
    - QgsVectorLayer: The generated random points layer.
    """
    random_pts_dir = os.path.join(base_dir, f"Point_data/Random_points/{year}/{month_name}")
    os.makedirs(random_pts_dir, exist_ok=True)
    target_crs = QgsCoordinateReferenceSystem('EPSG:3347')
    output_path = os.path.join(random_pts_dir, f"Random_NoFire_{month_name}_{year}.shp")

    # === Validate BC boundary layer
    if not bc_boundary_layer_3347.isValid():
        print("❌ BC boundary layer is not valid.")
        return None

    # === Get boundary extent for sampling
    extent = bc_boundary_layer_3347.extent()
    xmin, xmax, ymin, ymax = extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum()

    # === Transformer for lat/lon conversion
    transformer = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)

    # === Create random point layer
    layer = QgsVectorLayer(f"Point?crs={target_crs.authid()}", f"Random_NoFire_{month_name}_{year}", "memory")
    provider = layer.dataProvider()
    
    common_fields = [
    QgsField("Latitude", QVariant.Double),
    QgsField("Longitude", QVariant.Double),
    QgsField("Month", QVariant.String),
    QgsField("Year", QVariant.Int),
    QgsField("Fire", QVariant.Int)
    ]
    
    provider.addAttributes(common_fields)
    layer.updateFields()

    # === Generate points inside boundary
    features = []
    tries = 0
    max_tries = 10000

    while len(features) < non_fire_count and tries < max_tries:
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        point = QgsPointXY(x, y)
        geom = QgsGeometry.fromPointXY(point)

        if any(f.geometry().contains(geom) for f in bc_boundary_layer_3347.getFeatures()):
            lon, lat = transformer.transform(x, y)
            feat = QgsFeature()
            feat.setGeometry(geom)
            feat.setAttributes([lat, lon, month_name, year, 0])
            features.append(feat)
        tries += 1

    provider.addFeatures(features)
    layer.updateExtents()

    if len(features) < non_fire_count:
        print(f"⚠️ Only generated {len(features)} of {non_fire_count} points after {max_tries} attempts.")
    else:
        print(f"✅ Successfully generated {non_fire_count} no-fire points for {month_name} {year}.")

    QgsVectorFileWriter.writeAsVectorFormat(layer, output_path, "UTF-8", target_crs, "ESRI Shapefile")
    print(f"✅ Random no-fire layer saved: {output_path}")

    return layer, common_fields, transformer, target_crs





# === Clean hotspot layer
def rebuild_hotspot_clean_copy(hotspot_layer_3347, common_fields):
    
    if hotspot_layer_3347 is None:
        print("ℹ️ No fire points to clean.")
        return None

    if not hotspot_layer_3347.isValid():
        print("❌ Hotspot layer is not valid.")
        return None

    print("🔁 Creating clean memory copy...")

    # === Create memory layer with correct schema ===
    cleaned_layer = QgsVectorLayer(f"Point?crs={hotspot_layer_3347.crs().authid()}", "Cleaned_Hotspot", "memory")
    provider = cleaned_layer.dataProvider()
    provider.addAttributes(common_fields)
    cleaned_layer.updateFields()

    # === Detect LAT/LON and REP_DATE field names ===
    original_fields = hotspot_layer_3347.fields().names()

    lat_name = 'LAT' if 'LAT' in original_fields else 'lat'
    lon_name = 'LON' if 'LON' in original_fields else 'lon'
    date_name = 'REP_DATE' if 'REP_DATE' in original_fields else 'rep_date'

    for feat in hotspot_layer_3347.getFeatures():
        try:
            rep_date = feat[date_name]
            if "/" in rep_date:
                dt = datetime.strptime(rep_date, "%Y/%m/%d %H:%M:%S.%f")
            else:
                dt = datetime.strptime(rep_date, "%Y-%m-%d %H:%M:%S")

            month = dt.strftime('%B')
            year = dt.year
        except Exception as e:
            print(f"⚠️ Failed date parsing for feature ID {feat.id()}: {e}")
            month, year = '', 0

        lat = feat[lat_name]
        lon = feat[lon_name]

        new_feat = QgsFeature()
        new_feat.setGeometry(feat.geometry())
        new_feat.setAttributes([lat, lon, month, year, 1])  # Fire=1
        provider.addFeature(new_feat)

    cleaned_layer.updateExtents()
    print("✅ Cleaned hotspot layer built and loaded.")
    return cleaned_layer





def reorder_hotspots(cleaned_layer, layer, transformer, common_fields):
    # If no cleaned fire points layer is provided, return None
    if cleaned_layer is None:
        print("ℹ️ No fire points to reorder.")
        return None

    reordered_hotspot = QgsVectorLayer(f'Point?crs={cleaned_layer.crs().authid()}', "Reordered_Hotspot", "memory")
    provider_hotspot = reordered_hotspot.dataProvider()
    provider_hotspot.addAttributes(common_fields)
    reordered_hotspot.updateFields()
    
    # Add fire features
    for feat in cleaned_layer.getFeatures():
        f = QgsFeature()
        f.setGeometry(feat.geometry())
        f.setAttributes([
            feat['Latitude'], feat['Longitude'], feat['Month'], feat['Year'], feat['Fire']
        ])
        provider_hotspot.addFeature(f)
    
    reordered_hotspot.updateExtents()
    print("✅ Reordered hotspot layer created.")

    # Recalculate lat/lon for no-fire layer
    with edit(layer):
        lat_idx = layer.fields().indexOf("Latitude")
        lon_idx = layer.fields().indexOf("Longitude")
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom and not geom.isMultipart():
                x, y = geom.asPoint()
                lon, lat = transformer.transform(x, y)
                feature.setAttribute(lat_idx, lat)
                feature.setAttribute(lon_idx, lon)
                layer.updateFeature(feature)

    print("✅ Recalculated WGS84 coordinates for non-fire layer.")
    return reordered_hotspot

#-- MERGE fire and non-fire data points
def merge_data_points (month_name, year, month_num, reordered_hotspot, layer, common_fields, target_crs):
    # Use CRS from existing layer (either fire or non-fire)
    base_crs = reordered_hotspot.crs().authid() if reordered_hotspot else layer.crs().authid()
    merged_output_folder = os.path.join(base_dir, f"Point_data/Merged/{year}/{month_name}")
    os.makedirs(merged_output_folder, exist_ok=True)
    merged_path = os.path.join(merged_output_folder, f"Merged_Fire_NoFire_{month_name}_{year}.shp")
    merged_layer = QgsVectorLayer(f'Point?crs={base_crs}', "Merged", "memory")
    merged_provider = merged_layer.dataProvider()
    merged_provider.addAttributes(common_fields)
    merged_layer.updateFields()

    # Add both layers' features
    for src_layer in [reordered_hotspot, layer]:
        if src_layer is not None:
            for feat in src_layer.getFeatures():
                f = QgsFeature()
                f.setGeometry(feat.geometry())
                f.setAttributes([
                    feat['Latitude'], feat['Longitude'], feat['Month'], feat['Year'], feat['Fire']
                ])
                merged_provider.addFeature(f)

    merged_layer.updateExtents()
    QgsVectorFileWriter.writeAsVectorFormat(merged_layer, merged_path, "UTF-8", target_crs, "ESRI Shapefile")

    if os.path.exists(merged_path):
        print(f"✅ Merged dataset saved to: {merged_path}")
    else:
        print("❌ Failed to write merged shapefile.")
    
    return merged_layer

# == POINT SAMPLING TIME!
def point_sampling(month_name, year, merged_layer, bc_climate_filled):
    # === Input layer and output path ===
    sampled_output_folder = os.path.join(base_dir, f"Point_data/Sampled/{year}/{month_name}")
    os.makedirs(sampled_output_folder, exist_ok=True)
    sampled_output_path = os.path.join(sampled_output_folder, f"Sampled_Points_{month_name}{year}.shp")
    input_points = merged_layer  # your already prepared point layer

    # === Rasters and field names ===
    raster_info = [
        (bc_climate_filled, 1, 'u10_wind'),
        (bc_climate_filled, 2, 'v10_wind'),
        (bc_climate_filled, 3, 'dew_temp_2m'),
        (bc_climate_filled, 4, 'temp_2m'),
        (bc_climate_filled, 5, 'tot_precip'),
        (bc_climate_filled, 6, 'lai_high'),
        (os.path.join(base_dir, 'National_FBP_Fueltypes_version2014b/BC_fuel_type_epsg3347.tif'), 1, 'Fuel_Type')
    ]


    # === Copy of the input layer to keep results in memory ===
    temp_layer = QgsVectorLayer("Point?crs=" + input_points.crs().authid(), "Sampled Points", "memory")
    temp_layer.dataProvider().addAttributes(input_points.fields())
    temp_layer.updateFields()

    # Add original points
    temp_layer.dataProvider().addFeatures(list(input_points.getFeatures()))

    # === Run sampling one raster at a time ===
    for raster_path, band, output_field in raster_info:
        if not os.path.exists(raster_path):
            print(f"❌ Missing raster: {raster_path}")
            continue

        print(f"🔄 Sampling from: {output_field} (Band {band})")

        result = processing.run("qgis:rastersampling", {
            'INPUT': temp_layer,
            'RASTERCOPY': raster_path,
            'RASTER_BAND': band,
            'COLUMN_PREFIX': 'tmp_',
            'OUTPUT': 'memory:'
        })

        sampled = result['OUTPUT']

        # Rename the temp_... field to the intended name
        temp_fields = sampled.fields()
        temp_field_names = [f.name() for f in temp_fields if f.name().startswith('tmp_')]

        if not temp_field_names:
            print(f"⚠️ No field found for raster {output_field}")
            return

        temp_field = temp_field_names[0]
        idx = sampled.fields().indexOf(temp_field)

        with edit(sampled):
            # Rename the temporary field
            sampled.renameAttribute(idx, output_field)
            sampled.updateFields()

            # Delete all other temp_ fields
            for field in sampled.fields():
                name = field.name()
                if name.startswith('tmp_') and name != output_field:
                    sampled.deleteAttribute(sampled.fields().indexOf(name))
                    temp_indices = [sampled.fields().indexOf(f.name()) for f in sampled.fields() if f.name().startswith('tmp_') and f.name() != output_field]
                    for idx in sorted(temp_indices, reverse=True):
                        sampled.deleteAttribute(idx)
            sampled.updateFields()

        # Carry forward for next sampling
        temp_layer = sampled

    QgsVectorFileWriter.writeAsVectorFormat(temp_layer, sampled_output_path, 'UTF-8', temp_layer.crs(), 'ESRI Shapefile')
    
    return temp_layer, raster_info


# -- Clean missing values
def clean_sampled_layer(temp_layer, raster_info):
    # === Filter out features with nulls in any raster fields ===
    clean_layer = QgsVectorLayer(f"Point?crs={temp_layer.crs().authid()}", "Cleaned Sampled Points", "memory")
    clean_provider = clean_layer.dataProvider()
    clean_provider.addAttributes(temp_layer.fields())
    clean_layer.updateFields()

    # List of relevant raster field names to check for nulls
    fields_to_check = [f[2] for f in raster_info] 

    # Ensure fields exist before checking for nulls
    valid_fields = [f.name() for f in temp_layer.fields()]
    fields_to_check = [f[2] for f in raster_info if f[2] in valid_fields]

    removed = 0
    for feat in temp_layer.getFeatures():
        if any(feat[field] in [None, '', -9999] or isinstance(feat[field], float) and math.isnan(feat[field]) for field in fields_to_check):
            removed += 1
            continue

        clean_feat = QgsFeature()
        clean_feat.setGeometry(feat.geometry())
        clean_feat.setAttributes(feat.attributes())
        clean_provider.addFeature(clean_feat)

    print(f"✅ Filtered out {removed} features with missing values.")
    print(f"📦 Remaining features: {clean_layer.featureCount()}")

    QgsProject.instance().addMapLayer(clean_layer)
    
    return clean_layer


# === Save as CSV ===
def save_point_file(year, month_name, clean_layer, base_dir):
    # === Save as CSV ===
    csv_output_path = os.path.join(base_dir, f"Point_data/Sampled/{year}/Cleaned_Sampled_Points_{month_name}{year}.csv")
    QgsVectorFileWriter.writeAsVectorFormat(
        clean_layer, csv_output_path, "UTF-8", clean_layer.crs(), "CSV", layerOptions=['GEOMETRY=AS_XY']
    )

    if os.path.exists(csv_output_path):
        print(f"✅ Cleaned attribute table saved to CSV: {csv_output_path}")
    else:
        print("❌ Failed to save CSV file.")
        
    return csv_output_path

# --- Loop through years and months
for year in range(start_year, end_year + 1):
    print(f"Processing {year}...")
    # Load hotspot data
    hotspot_layer = get_hotspots(year)
    
    # Loop through months
    for month_num, month_name in month_words.items():
        # Filter by month and clip to BC
        fire_points = get_monthly_hotspot_data(month_num, month_name, year, hotspot_layer)

        if fire_points is None:
            print(f"⚠️ Skipping {month_name} {year} — no valid hotspot data.")
            fire_counts[(year, month_name)] = 0
            non_fire_counts[(year, month_name)] = None
            continue
        
        # Get number of fire points and save them
        fire_count = len(fire_points)
        fire_counts[(year, month_name)] = fire_count
        
        # Temporarily store None if there are no fire points; will compute average later
        if fire_count > 0:
            non_fire_counts[(year, month_name)] = fire_count
        else:
            non_fire_counts[(year, month_name)] = None
            
# Compute average fire count for months where fire occurred
valid_fire_counts = [count for count in fire_counts.values() if count > 0]
avg_fire_count = int(round(sum(valid_fire_counts) / len(valid_fire_counts))) if valid_fire_counts else 400

# Update non-fire counts for months with 0 fire points
for (year, month_name), count in non_fire_counts.items():
    if count is None:
        non_fire_counts[(year, month_name)] = avg_fire_count

all_csv_paths = []

# Second pass: process each year/month using determined non-fire counts
for year in range(start_year, end_year + 1):
    print(f"Processing {year}...")
    # Load hotspot data
    hotspot_layer = get_hotspots(year)
    
    # Loop through months
    for month_num, month_name in month_words.items():
        # Filter by month and clip to BC
        fire_points = get_monthly_hotspot_data(month_num, month_name, year, hotspot_layer)
        
        if fire_points is None:
            print(f"ℹ️ No fire points found for {month_name} {year}. Proceeding with non-fire data only.")
        
        # Step 2: Generate non-fire points based on logic above
        non_fire_count = non_fire_counts[(year, month_name)]
        layer, common_fields, transformer, target_crs = gen_non_fire_points(year, month_name, non_fire_count)
        
        # Clean hotspot layer
        cleaned_layer = rebuild_hotspot_clean_copy(fire_points, common_fields)
        
        # Reorder hotspot data
        reordered_hotspot = reorder_hotspots(cleaned_layer, layer, transformer, common_fields)
    
        # Step 3: Merge fire and non-fire points
        merged_layer = merge_data_points(month_name, year, month_num, reordered_hotspot, layer, common_fields, target_crs)

        # Get climate data
        bc_climate_filled = climate_extraction(year, month_name, month_num)
        
        # Sample using everything
        temp_layer, raster_info = point_sampling(month_name, year, merged_layer, bc_climate_filled)
        
        # Clean sampled layer
        clean_layer = clean_sampled_layer(temp_layer, raster_info)
        
        # Save CSV file
        csv_output_path = save_point_file(year, month_name, clean_layer, base_dir)
        all_csv_paths.append(csv_output_path)
        
        # Log progress
        print(f"Completed processing for {month_name} {year}")


        
# Combine all sampled monthly data into one CSV
combined_csv_path = os.path.join(base_dir, f"Point_data/Sampled/Combined_Sampled_Points_{start_year}-{end_year}.csv")
with open(combined_csv_path, 'w', newline='') as combined_file:
    writer = csv.writer(combined_file)
    header_written = False

    for path in all_csv_paths:
        if os.path.exists(path):
            with open(path, 'r') as infile:
                reader = csv.reader(infile)
                header = next(reader)
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)

print(f"All data processing complete. Combined CSV saved as 'Complete_Sampled_Data_{start_year}-{end_year}.csv'")