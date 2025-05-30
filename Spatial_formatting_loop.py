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
                       QgsCoordinateReferenceSystem, QgsProcessingProvider, QgsFields
)
from processing.algs.gdal.GdalAlgorithmProvider import GdalAlgorithmProvider
import csv
from collections import defaultdict
from dateutil import parser
import logging



# Initialize Processing framework
Processing.initialize()

# Register built-in QGIS and GDAL providers
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())
QgsApplication.processingRegistry().addProvider(GdalAlgorithmProvider())

logging.info("✅ GDAL Processing tools enabled.")

# --- Load Canada provinces shapefile
base_dir = 'C:/Users/tdoa2/Downloads/Spatial data cleaning/'
canada_provinces_path = os.path.join(base_dir, 'Map_of_Canada/lpr_000b16a_e.shp')
canada_layer = QgsVectorLayer(canada_provinces_path, 'Canada Provinces', 'ogr')

if not canada_layer.isValid():
    logging.info("❌ Canada provinces layer failed to load.")
else:
    logging.info("✅ Canada provinces layer created.")

# Logging system
log_path = os.path.join(base_dir, 'script_logs.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')


# --- Extract British Columbia boundary
bc_output_path = os.path.join(base_dir, 'Map_of_Canada/BC_boundary.shp')

processing.run("native:extractbyattribute", {
    'INPUT': canada_layer,
    'FIELD': 'PREABBR', 
    'OPERATOR': 0,  # '='
    'VALUE': 'B.C.', 
    'OUTPUT': bc_output_path
})
logging.info("✅ BC boundary extracted and saved to:", bc_output_path)

# --- Load BC layer
bc_boundary_layer = QgsVectorLayer(bc_output_path, 'BC Boundary', 'ogr')
if bc_boundary_layer.isValid():
    logging.info("✅ BC Boundary layer created.")
else:
    logging.info("❌ Failed to load BC Boundary layer.")

# === STEP: Reproject BC Boundary to EPSG:3347 ===
reprojected_bc_boundary = os.path.join(base_dir, 'Map_of_Canada/BC_boundary_epsg3347.shp')

processing.run("native:reprojectlayer", {
    'INPUT': bc_output_path,
    'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:3347'),
    'OUTPUT': reprojected_bc_boundary
})
logging.info("✅ Reprojected BC boundary to EPSG:3347")

# Load reprojected BC boundary
bc_boundary_layer_3347 = QgsVectorLayer(reprojected_bc_boundary, 'BC Boundary (EPSG:3347)', 'ogr')
if bc_boundary_layer_3347.isValid():
    QgsProject.instance().addMapLayer(bc_boundary_layer_3347)
    logging.info("✅ Reprojected BC boundary layer created successfully.")
else:
    logging.info("❌ Failed to load reprojected BC boundary layer.")

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
logging.info("✅ Temporary clipped fuel raster created.")

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
    logging.info("✅ Reprojected raster to EPSG:3347 successfully.")

    reprojected_fuel_layer = QgsRasterLayer(final_clipped_raster, "BC Fuel Type (EPSG:3347)")
    if reprojected_fuel_layer.isValid():
        logging.info("✅ Reprojected raster added to project.")
        QgsProject.instance().addMapLayer(reprojected_fuel_layer)
    else:
        logging.info("❌ Failed to load reprojected raster.")
else:
    logging.info(f"❌ File not found: {clipped_raster_temp}")
    
    
    
# === PARAMETERS
start_year = 2000
end_year = 2024

# --- Months dictionary
month_words = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


# -- Total fires
fire_counts = {}  # Store fire point count for each (year, month)
non_fire_counts = {} # Store planned non-fire point count
yearly_fire_counts = defaultdict(list)  # {year: [monthly fire counts]}

# Store all reprojected layers so we don't recompute
reprojected_hotspot_layers = {}

# BC Boundary features
spatial_index = QgsSpatialIndex(bc_boundary_layer_3347.getFeatures())
boundary_features = {f.id(): f for f in bc_boundary_layer_3347.getFeatures()}



# == FUNCTIONS
# -- Get fire hotspots files
def get_hotspots(year):
    # Load Hotspot shapefile
    hotspot_path = os.path.join(base_dir, f"Point_data/Hotspot data/{year}_hotspots/{year}_hotspots.shp")
    hotspot_layer = QgsVectorLayer(hotspot_path, f"Hotspots {year}", 'ogr')
    
    if not hotspot_layer.isValid():
        logging.info("❌ Hotspot shapefile failed to load.")
        return
    else:
        logging.info(f"✅ {year} Hotspot layer created.")
    return hotspot_layer
    
    
def reproject_hotspot_layer(hotspot_layer, year):
    reprojected_hotspot_folder = os.path.join(base_dir, f"Point_data/Hotspot data/{year}_hotspots/Reprojected_hotspot_files")
    os.makedirs(reprojected_hotspot_folder, exist_ok=True)
    reprojected_hotspot_path = os.path.join(reprojected_hotspot_folder, f"{year}_reprojected.shp")

    if not os.path.exists(reprojected_hotspot_path):
        processing.run("native:reprojectlayer", {
            'INPUT': hotspot_layer,
            'TARGET_CRS': QgsCoordinateReferenceSystem('EPSG:3347'),
            'OUTPUT': reprojected_hotspot_path
        })
        logging.info(f"✅ Reprojected hotspot layer for {year} saved.")
    else:
        logging.info(f"ℹ️ Reprojected hotspot file for {year} already exists.")

    reprojected_hotspot_layer = QgsVectorLayer(reprojected_hotspot_path, f"Reprojected {year} Hotspots", 'ogr')
    if not isinstance(reprojected_hotspot_layer, QgsVectorLayer):
        logging.error("❌ Input is not a vector layer. Make sure to pass a QgsVectorLayer, not a raster.")
        return None
    return reprojected_hotspot_layer


def get_monthly_hotspot_data(month_num, month_name, year, reprojected_hotspot_layer):
    # Detect date field
    field_names = [field.name() for field in reprojected_hotspot_layer.fields()]
    if 'REP_DATE' in field_names:
        date_field = 'REP_DATE'
    elif 'rep_date' in field_names:
        date_field = 'rep_date'
    else:
        logging.info(f"⚠️ {year} hotspot layer: No REP_DATE or rep_date field.")
        return None

    # Filter features for the given month and year
    selected_features = []
    for feature in reprojected_hotspot_layer.getFeatures():
        try:
            raw_date = feature[date_field]
            if not raw_date:
                continue
            date_obj = parser.parse(str(raw_date))
            if date_obj.year == year and date_obj.month == month_num:
                selected_features.append(feature)
        except Exception as e:
            logging.info(f"⚠️ Skipping feature with bad date: {raw_date}, error: {e}")

    if not selected_features:
        logging.info(f"⚠️ {month_name} {year}: No features matched the date.")
        return None

    # Step 4: Rebuild geometries and enforce CRS alignment
    new_features = []
    for feat in selected_features:
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        new_feat = QgsFeature()
        new_feat.setFields(reprojected_hotspot_layer.fields())
        new_feat.setGeometry(QgsGeometry(geom))  # Deep copy
        for field in reprojected_hotspot_layer.fields():
            new_feat.setAttribute(field.name(), feat[field.name()])
        new_features.append(new_feat)

    # Step 5: Create memory layer in EPSG:3347, not EPSG:4326
    temp_layer = QgsVectorLayer("Point?crs=EPSG:3347", f"{year}_{month_name}_Hotspots", "memory")
    temp_layer.dataProvider().addAttributes(reprojected_hotspot_layer.fields())
    temp_layer.updateFields()
    temp_layer.dataProvider().addFeatures(new_features)
    temp_layer.updateExtents()

    if temp_layer.isValid():
        logging.info(f"ℹ️ {month_name} {year}: Filtered layer has {temp_layer.featureCount()} features.")
    else:
        logging.info(f"❌ {month_name} {year} layer is not valid.")
        return None

    # Step 6: Clip to BC using EPSG:3347 boundary layer
    if not bc_boundary_layer_3347.isValid():
        logging.info("❌ BC boundary layer is not valid.")
        return None

    result = processing.run("native:clip", {
        'INPUT': temp_layer,
        'OVERLAY': bc_boundary_layer_3347,
        'OUTPUT': 'memory:'
    })
    clipped_layer = result['OUTPUT']

    if clipped_layer.featureCount() == 0:
        logging.info(f"⚠️ {month_name} {year}: Clipped layer has no features.")
        return None

    clipped_layer.setName(f"Hotspots {month_name} {year} EPSG:3347")
    clipped_layer.updateExtents()  # Explicit extent update again

    logging.info(f"✅ {month_name} {year}: Final layer has {clipped_layer.featureCount()} features.")
    return clipped_layer


# Get climate data
def get_climate_raster_path(year, month_name):
    band_folder = os.path.join(
        base_dir,
        f"climate_data/GRIB_climate_data/{year}/{month_name}/Filled_Bands_{month_name}_{year}"
    )
    climate_stack = os.path.join(band_folder, f"{month_name}_{year}_Filled_Stacked_Climate.tif")
    if not os.path.exists(climate_stack):
        logging.info(f"❌ Missing climate raster: {climate_stack}")
        return None, None
    return climate_stack



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
        logging.info("❌ BC boundary layer is not valid.")
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

        ids = spatial_index.intersects(geom.boundingBox())
        if any(boundary_features[i].geometry().contains(geom) for i in ids):
            lon, lat = transformer.transform(x, y)
            feat = QgsFeature()
            feat.setGeometry(geom)
            feat.setAttributes([lat, lon, month_name, year, 0])
            features.append(feat)
        tries += 1

    provider.addFeatures(features)
    layer.updateExtents()

    if len(features) < non_fire_count:
        logging.info(f"⚠️ Only generated {len(features)} of {non_fire_count} points after {max_tries} attempts.")
    else:
        logging.info(f"✅ Successfully generated {non_fire_count} no-fire points for {month_name} {year}.")

    QgsVectorFileWriter.writeAsVectorFormat(layer, output_path, "UTF-8", target_crs, "ESRI Shapefile")
    logging.info(f"✅ Random no-fire layer saved: {output_path}")

    return layer, common_fields, transformer, target_crs






# === Clean hotspot layer
def rebuild_hotspot_clean_copy(clipped_layer, common_fields):
    
    if clipped_layer is None:
        logging.info("ℹ️ No fire points to clean.")
        return None

    if not clipped_layer.isValid():
        logging.info("❌ Hotspot layer is not valid.")
        return None

    logging.info("🔁 Creating clean memory copy...")

    # === Create memory layer with correct schema ===
    cleaned_layer = QgsVectorLayer(f"Point?crs={clipped_layer.crs().authid()}", "Cleaned_Hotspot", "memory")
    provider = cleaned_layer.dataProvider()
    provider.addAttributes(common_fields)
    cleaned_layer.updateFields()

    # === Detect LAT/LON and REP_DATE field names ===
    original_fields = clipped_layer.fields().names()

    lat_name = 'LAT' if 'LAT' in original_fields else 'lat'
    lon_name = 'LON' if 'LON' in original_fields else 'lon'
    date_name = 'REP_DATE' if 'REP_DATE' in original_fields else 'rep_date'

    for feat in clipped_layer.getFeatures():
        try:
            rep_date = feat[date_name]
            if "/" in rep_date:
                dt = datetime.strptime(rep_date, "%Y/%m/%d %H:%M:%S.%f")
            else:
                dt = datetime.strptime(rep_date, "%Y-%m-%d %H:%M:%S")

            month = dt.strftime('%B')
            year = dt.year
        except Exception as e:
            logging.info(f"⚠️ Failed date parsing for feature ID {feat.id()}: {e}")
            month, year = '', 0

        lat = feat[lat_name]
        lon = feat[lon_name]

        new_feat = QgsFeature()
        new_feat.setGeometry(feat.geometry())
        new_feat.setAttributes([lat, lon, month, year, 1])  # Fire=1
        provider.addFeature(new_feat)

    cleaned_layer.updateExtents()
    logging.info("✅ Cleaned hotspot layer built and loaded.")
    return cleaned_layer





def reorder_hotspots(cleaned_layer, layer, transformer, common_fields):
    # If no cleaned fire points layer is provided, return None
    if cleaned_layer is None:
        logging.info("ℹ️ No fire points to reorder.")
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
    logging.info("✅ Reordered hotspot layer created.")

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

    logging.info("✅ Recalculated WGS84 coordinates for non-fire layer.")
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
        logging.info(f"✅ Merged dataset saved to: {merged_path}")
    else:
        logging.info("❌ Failed to write merged shapefile.")
    
    return merged_layer


def rename_climate_fields(layer, rename_map):
    """Renames Band_x fields to meaningful climate/fuel variable names."""
    if not layer.isEditable():
        layer.startEditing()

    fields = layer.fields()
    for old_name, new_name in rename_map.items():
        idx = fields.indexOf(old_name)
        if idx != -1:
            logging.info(f"🔤 Renaming {old_name} ➜ {new_name}")
            layer.renameAttribute(idx, new_name)
        else:
            logging.info(f"⚠️ Field {old_name} not found in layer")

    layer.commitChanges()


# == POINT SAMPLING TIME!
def point_sampling(month_name, year, merged_layer, climate_stack):
    # === Output setup ===
    sampled_output_folder = os.path.join(base_dir, f"Point_data/Sampled/{year}/{month_name}")
    os.makedirs(sampled_output_folder, exist_ok=True)
    sampled_output_path = os.path.join(sampled_output_folder, f"Sampled_Points_{month_name}{year}.shp")

    # === Temporary paths ===
    temp_input_path = os.path.join(sampled_output_folder, "temp_input_points.shp")
    temp_climate_sampled_path = os.path.join(sampled_output_folder, "temp_climate_sampled.shp")

    # === Save input merged layer to shapefile ===
    QgsVectorFileWriter.writeAsVectorFormat(merged_layer, temp_input_path, "UTF-8", merged_layer.crs(), "ESRI Shapefile")

    # === Load climate raster layer as QgsRasterLayer objects ===
    climate_raster = QgsRasterLayer(climate_stack, f"Climate Raster_{month_name}_{year}")
    if not climate_raster.isValid():
        logging.error(f"❌ Invalid climate raster: {climate_stack}")
        return None, None, sampled_output_folder

    
    if not reprojected_fuel_layer.isValid():
        logging.error(f"❌ Invalid fuel raster: {fuel_raster_path}")
        return None, None, sampled_output_folder

    # === Sample climate stack ===
    processing.run("native:rastersampling", {
        'INPUT': temp_input_path,
        'RASTERCOPY': climate_raster,
        'COLUMN_PREFIX': '',
        'OUTPUT': temp_climate_sampled_path
    })

    # === Sample fuel raster ===
    processing.run("native:rastersampling", {
        'INPUT': temp_climate_sampled_path,
        'RASTERCOPY': reprojected_fuel_layer,
        'COLUMN_PREFIX': 'Fuel_',
        'OUTPUT': sampled_output_path
    })

    logging.info(f"✅ Sampled points saved to: {sampled_output_path}")

    # === Rename fields ===
    sampled_layer = QgsVectorLayer(sampled_output_path, "Sampled Points", "ogr")
    if not sampled_layer.isValid():
        logging.error(f"❌ Failed to load sampled layer: {sampled_output_path}")
        return None, None, sampled_output_folder

    
    rename_map = {
        '1': 'u10_wind',
        '2': 'v10_wind',
        '3': 'dew_temp_2m',
        '4': 'temp_2m',
        '5': 'tot_precip',
        '6': 'lai_high',
        'Fuel_1': 'Fuel_Type'
    }

    rename_climate_fields(sampled_layer, rename_map)

    return sampled_layer, None, sampled_output_folder




# -- Clean missing values
def clean_sampled_layer(sampled_layer):
    """Removes features with missing climate or fuel data."""
    crs = sampled_layer.crs().authid()
    clean_layer = QgsVectorLayer(f"Point?crs={crs}", "Cleaned Sampled Points", "memory")
    clean_provider = clean_layer.dataProvider()
    clean_provider.addAttributes(sampled_layer.fields())
    clean_layer.updateFields()

    # Fields expected after renaming
    fields_to_check = [
        'u10_wind', 'v10_wind', 'dew_temp_2m', 'temp_2m',
        'tot_precip', 'lai_high', 'Fuel_Type'
    ]

    # Check which expected fields exist
    valid_fields = [f.name() for f in sampled_layer.fields()]
    fields_to_check = [f for f in fields_to_check if f in valid_fields]

    removed = 0
    for feat in sampled_layer.getFeatures():
        has_null = any(
            feat[field] in [None, '', -9999] or
            (isinstance(feat[field], float) and math.isnan(feat[field]))
            for field in fields_to_check
        )
        if has_null:
            removed += 1
            continue

        clean_feat = QgsFeature()
        clean_feat.setGeometry(feat.geometry())
        clean_feat.setAttributes(feat.attributes())
        clean_provider.addFeature(clean_feat)

    logging.info(f"✅ Filtered out {removed} features with missing values.")
    logging.info(f"📦 Remaining features: {clean_layer.featureCount()}")

    QgsProject.instance().addMapLayer(clean_layer)

    return clean_layer


# === Save as CSV ===
def save_point_file(year, month_name, clean_layer, sampled_output_folder):
    # === Save as CSV ===
    csv_output_path = os.path.join(sampled_output_folder, f"Cleaned_Sampled_Points_{month_name}{year}.csv")
    QgsVectorFileWriter.writeAsVectorFormat(
        clean_layer, csv_output_path, "UTF-8", clean_layer.crs(), "CSV", layerOptions=['GEOMETRY=AS_XY']
    )

    if os.path.exists(csv_output_path):
        logging.info(f"✅ Cleaned attribute table saved to CSV: {csv_output_path}")
    else:
        logging.info("❌ Failed to save CSV file.")
        
    return csv_output_path





#== LOOP THROUGH DIFFERENT YEARS AND MONTHS
for year in range(start_year, end_year + 1):
    logging.info(f"Processing {year}...")
    # Load hotspot data
    hotspot_layer = get_hotspots(year)
    
    # Reproject hotspot layer
    reprojected_hotspot_layer = reproject_hotspot_layer(hotspot_layer, year)  # pass year explicitly
    reprojected_hotspot_layers[year] = reprojected_hotspot_layer
    
    # Loop through months
    for month_num, month_name in month_words.items():
        reprojected_hotspot_layer = reprojected_hotspot_layers[year]
        # Filter by month and clip to BC
        fire_points = get_monthly_hotspot_data(month_num, month_name, year, reprojected_hotspot_layer)

        if fire_points is None:
            logging.info(f"⚠️ Skipping {month_name} {year} — no valid hotspot data.")
            fire_counts[(year, month_name)] = 0
            non_fire_counts[(year, month_name)] = None
            continue
        
        # Get number of fire points and save them
        fire_count = len(fire_points)
        fire_counts[(year, month_name)] = fire_count
        
        
        # Temporarily store None if there are no fire points; will compute average later
        if fire_count > 0:
            non_fire_counts[(year, month_name)] = fire_count
            yearly_fire_counts[year].append(fire_count)
        else:
            non_fire_counts[(year, month_name)] = None

# Compute yearly average fire counts
yearly_avg_fire = {
    year: int(round(sum(counts) / len(counts))) if counts else 400
    for year, counts in yearly_fire_counts.items()
}


# Update non-fire counts: use fire count if available, otherwise use yearly average
for (year, month_name), count in fire_counts.items():
    if count > 0:
        non_fire_counts[(year, month_name)] = count
    else:
        non_fire_counts[(year, month_name)] = yearly_avg_fire.get(year, 400)

all_csv_paths = []

# Second pass: process each year/month using determined non-fire counts
for year in range(start_year, end_year + 1):
    logging.info(f"Processing {year}...")
    # Load hotspot data
    hotspot_layer = get_hotspots(year)
    
    # Loop through months
    for month_num, month_name in month_words.items():
        # Filter by month and clip to BC
        reprojected_hotspot_layer = reprojected_hotspot_layers[year]
        fire_points = get_monthly_hotspot_data(month_num, month_name, year, reprojected_hotspot_layer)
        
        if fire_points is None:
            logging.info(f"ℹ️ No fire points found for {month_name} {year}. Proceeding with non-fire data only.")
        
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
        climate_stack = get_climate_raster_path(year, month_name)
        
        
        # Sample using everything
        sampled_layer, _, sampled_output_folder = point_sampling(month_name, year, merged_layer, climate_stack)
        
        # Clean sampled layer
        clean_layer = clean_sampled_layer(sampled_layer)
        
        # Save CSV file
        csv_output_path = save_point_file(year, month_name, clean_layer, sampled_output_folder)
        all_csv_paths.append(csv_output_path)
        
        # Log progress
        logging.info(f"Completed processing for {month_name} {year}")


        
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

logging.info(f"All data processing complete. Combined CSV saved as 'Complete_Sampled_Data_{start_year}-{end_year}.csv'")