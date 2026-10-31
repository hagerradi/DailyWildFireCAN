import subprocess
from pathlib import Path
import rasterio
import argparse
import os

from configs import settings
from data_preparation.data_configs.data_settings import METER_NAD_CRS

def process_disturbance_raster_gdal_90m(input_folder: str, output_folder: str, filename: str, target_crs: str = None):
    """
    Resamples a single DISTURBANCE .tif file to 90m using GDAL mode resampling.
    Skips processing if the target file already exists in the output folder.
    """

    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # We need temporary files for the GDAL sandwich
    temp_masked_path = output_dir / f"temp_masked_{filename}"
    temp_warped_path = output_dir / f"temp_warped_{filename}"
    
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
    
    # Execution
    try:

        # STEP 1: Mask 0s to 255 (Background -> NoData)
        calc_cmd_1 = [
            "gdal_calc.py",
            "-A", str(input_path),
            "--outfile", str(temp_masked_path),
            "--calc", "A*(A>0) + 255*(A==0)",  # The math that moves 0 to 255
            "--NoDataValue", "255",
            "--type", "Byte",
            "--co", "COMPRESS=ZSTD",
            "--co", "TILED=YES",
            "--quiet"
        ]
        subprocess.run(calc_cmd_1, check=True)
        
        # STEP 2: Warp to 90m using Mode (Ignoring 255)
        warp_cmd = [
                "gdalwarp", 
                "-ot", "Byte",
                "-tr", f"{settings.PIXEL_SIZE}", f"{settings.PIXEL_SIZE}",
                "-r", "mode",
                "-wm", "4000",
                "-multi",
                "-srcnodata", "255",  # Explicitly tell GDAL to ignore 255
                "-dstnodata", "255",
                "-co", "COMPRESS=ZSTD",
                "-co", "TILED=YES",
                "-co", "NUM_THREADS=ALL_CPUS"
        ]
        if target_crs is not None:
            warp_cmd.extend(["-t_srs", target_crs])
                
        # Point to temp files instead of input_path/output_path
        warp_cmd.extend([str(temp_masked_path), str(temp_warped_path)]) 
        subprocess.run(warp_cmd, check=True)
        
        # STEP 3: Restore 255 to 0 (NoData -> Background)
        calc_cmd_2 = [
            "gdal_calc.py",
            "-A", str(temp_warped_path),
            "--outfile", str(output_path), # Final output goes to your actual desired path
            "--calc", "A*(A<255) + 0*(A==255)", # The math that moves 255 back to 0
            "--NoDataValue", "255", 
            "--type", "Byte",
            "--co", "COMPRESS=ZSTD",
            "--co", "TILED=YES",
            "--quiet"
        ]
        subprocess.run(calc_cmd_2, check=True)
        
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

    finally:
        # STEP 4: Cleanup Temporary Files
        if temp_masked_path.exists():
            os.remove(temp_masked_path)
        if temp_warped_path.exists():
            os.remove(temp_warped_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single raster to 90m for cluster parallelization.")
   
    parser.add_argument("--filename", type=str, required=True, help="Specific filename to process (e.g., 'disturbance_01.tif').")
    
    args = parser.parse_args()
    
    process_disturbance_raster_gdal_90m(
        input_folder=settings.DISTURBANCE_ORG_FOLDER,
        output_folder=settings.DISTURBANCE_FOLDER,
        filename=args.filename,
        target_crs=f"EPSG:{METER_NAD_CRS}"
    )