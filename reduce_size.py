import netCDF4 as nc
import os
import re

# Define the path to the parent directory where the dataset is located
input_directory = 'l1r_11_updated_10072024'
output_directory = 'reduced_nc_files'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Regex pattern to identify NetCDF files
pattern = re.compile(r'.*\.nc')

# Variables to keep
variables_to_keep = ['Epoch', 'ISS_Latitude', 'ISS_Longitude', 'Radiance']

# Compression settings
compression_level = 4  # Maximum compression

# Chunk size for Radiance
chunk_size = (25, 300, 300)

# Loop through all files in the directory
for root, dirs, files in os.walk(input_directory):
    for file in files:
        if pattern.match(file):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_directory, file)
            
            # Read the input NetCDF file
            with nc.Dataset(input_path, 'r') as src:
                # Create a new NetCDF file to save reduced data
                with nc.Dataset(output_path, 'w', format='NETCDF4') as dst:
                    # Copy global attributes
                    for attr in src.ncattrs():
                        dst.setncattr(attr, src.getncattr(attr))
                    
                    # Copy dimensions
                    for dim_name, dimension in src.dimensions.items():
                        dst.createDimension(dim_name, len(dimension) if not dimension.isunlimited() else None)
                    
                    # Copy only the selected variables
                    for var_name in variables_to_keep:
                        if var_name in src.variables:
                            var = src.variables[var_name]
                            
                            # Use specific chunk size for Radiance
                            if var_name == 'Radiance':
                                dst_var = dst.createVariable(
                                    var_name,
                                    var.dtype,
                                    var.dimensions,
                                    chunksizes=chunk_size,
                                    zlib=True,  # Enable compression
                                    complevel=compression_level  # Set compression level
                                )
                            else:
                                dst_var = dst.createVariable(
                                    var_name,
                                    var.dtype,
                                    var.dimensions,
                                    chunksizes=var.chunking(),  # Keep original chunking for other variables
                                    zlib=True,
                                    complevel=compression_level
                                )
                            
                            # Copy variable attributes
                            for attr in var.ncattrs():
                                dst_var.setncattr(attr, var.getncattr(attr))
                            
                            # Copy variable data
                            dst_var[:] = var[:]
            
            print(f"Reduced file saved: {output_path}")

print("File reduction complete.")
