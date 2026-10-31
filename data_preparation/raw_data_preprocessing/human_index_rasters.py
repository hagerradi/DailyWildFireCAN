import subprocess
from pathlib import Path
import rasterio
import argparse

from configs import settings
from data_preparation.data_configs.data_settings import METER_NAD_CRS

def process_single_hii_raster_gdal_90m(input_folder: str, output_folder: str, filename: str, target_crs: str = None):
    """
    Resamples a single HII .tif file to 90m using GDAL bilinear resampling.
    Skips processing if the target file already exists in the output folder.
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = input_dir / filename
    
    if not input_path.exists():
        print(f"Error: Input file not found -> {input_path}")
        return
        
    # Define output filename and check if it already exists
    output_filename = f"{input_path.stem}_90m{input_path.suffix}"
    output_path = output_dir / output_filename
    
    if output_path.exists():
        print(f"Skipping: Output already exists -> {output_filename}")
        return
        
    print(f"Processing: {filename} -> {output_filename}")

    # Extract NoData value dynamically AND print original raster info
    with rasterio.open(input_path) as src:
        input_nodata = src.nodata 
        print(f"\n--- Original Raster Info: {filename} ---")
        print(f"Data Type: {src.dtypes[0]}")
        print(f"NoData Value: {src.nodata}")
        print(f"Shape: {src.shape}")
        print(f"CRS: {src.crs}")
        print(f"Pixel Size (x, y): {src.res}")
        print("------------------------------------------\n")

    # Build the GDAL Command
    warp_cmd = [
        "gdalwarp", 
        "-ot", "Float32",
        "-tr", f"{settings.PIXEL_SIZE}", f"{settings.PIXEL_SIZE}",
        "-r", "bilinear",
        "-wm", "4000",
        "-multi",
        "-co", "COMPRESS=ZSTD",
        "-co", "PREDICTOR=3",
        "-co", "TILED=YES",
        "-co", "NUM_THREADS=ALL_CPUS",
        "-overwrite"
    ]
    
    # Handle on-the-fly reprojection
    if target_crs is not None:
        warp_cmd.extend(["-t_srs", target_crs])
    
    # Handle NoData mapping
    if input_nodata is not None:
        warp_cmd.extend(["-srcnodata", str(input_nodata), "-dstnodata", "nan"])
    
    # Append paths
    warp_cmd.extend([str(input_path), str(output_path)])
    
    # Execution
    try:
        result = subprocess.run(warp_cmd, check=True)

        print(f"Success: {output_filename} generated.")

        # Verify and print the newly generated raster info
        with rasterio.open(output_path) as out_src:
            print(f"\n--- New Raster Info: {output_filename} ---")
            print(f"Data Type: {out_src.dtypes[0]}")
            print(f"NoData Value: {out_src.nodata}")
            print(f"Shape: {out_src.shape}")
            print(f"CRS: {out_src.crs}")
            print(f"Pixel Size (x, y): {out_src.res}")
            print("----------------------------------------------\n")
    
    except subprocess.CalledProcessError as e:
        print(f"Error processing {filename}")
        print(f"GDAL Message: {e.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single raster to 90m for cluster parallelization.")
   
    parser.add_argument("--filename", type=str, required=True, help="Specific filename to process (e.g., 'hii_01.tif').")
    
    args = parser.parse_args()
    
    process_single_hii_raster_gdal_90m(
        input_folder=settings.HII_ORG_FOLDER,
        output_folder=settings.HII_FOLDER,
        filename=args.filename,
        target_crs=f"EPSG:{METER_NAD_CRS}"
    )