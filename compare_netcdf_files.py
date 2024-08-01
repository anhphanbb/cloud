import os
import numpy as np
from netCDF4 import Dataset

def compare_netcdf_files(file1_path, file2_path):
    with Dataset(file1_path, 'r') as nc1, Dataset(file2_path, 'r') as nc2:
        print(f"Comparing {file1_path} and {file2_path}:\n")

        # Compare global attributes
        print("Global Attributes:")
        attrs1 = nc1.ncattrs()
        attrs2 = nc2.ncattrs()
        compare_attributes(attrs1, attrs2, nc1, nc2)

        # Compare dimensions
        print("\nDimensions:")
        dims1 = nc1.dimensions
        dims2 = nc2.dimensions
        compare_dimensions(dims1, dims2)

        # Compare variables
        print("\nVariables:")
        vars1 = nc1.variables
        vars2 = nc2.variables
        compare_variables(vars1, vars2)

        # Compare data values
        print("\nData Values:")
        compare_data_values(vars1, vars2)

def compare_attributes(attrs1, attrs2, nc1, nc2):
    all_attrs = set(attrs1).union(set(attrs2))
    for attr in all_attrs:
        val1 = nc1.getncattr(attr) if attr in attrs1 else None
        val2 = nc2.getncattr(attr) if attr in attrs2 else None
        if val1 != val2:
            print(f" - Attribute {attr}:")
            print(f"   - File 1: {val1}")
            print(f"   - File 2: {val2}")

def compare_dimensions(dims1, dims2):
    all_dims = set(dims1.keys()).union(set(dims2.keys()))
    for dim in all_dims:
        len1 = len(dims1[dim]) if dim in dims1 else None
        len2 = len(dims2[dim]) if dim in dims2 else None
        if len1 != len2:
            print(f" - Dimension {dim}:")
            print(f"   - File 1: {len1}")
            print(f"   - File 2: {len2}")

def compare_variables(vars1, vars2):
    all_vars = set(vars1.keys()).union(set(vars2.keys()))
    for var in all_vars:
        if var not in vars1:
            print(f" - Variable {var} is missing in File 1")
        elif var not in vars2:
            print(f" - Variable {var} is missing in File 2")
        else:
            var1 = vars1[var]
            var2 = vars2[var]
            compare_variable_details(var, var1, var2)

def compare_variable_details(var_name, var1, var2):
    if var1.datatype != var2.datatype:
        print(f" - Variable {var_name} datatype differs:")
        print(f"   - File 1: {var1.datatype}")
        print(f"   - File 2: {var2.datatype}")
    
    if var1.dimensions != var2.dimensions:
        print(f" - Variable {var_name} dimensions differ:")
        print(f"   - File 1: {var1.dimensions}")
        print(f"   - File 2: {var2.dimensions}")
    
    attrs1 = var1.ncattrs()
    attrs2 = var2.ncattrs()
    compare_attributes(attrs1, attrs2, var1, var2)

def compare_data_values(vars1, vars2):
    all_vars = set(vars1.keys()).union(set(vars2.keys()))
    for var in all_vars:
        if var in vars1 and var in vars2:
            data1 = vars1[var][:]
            data2 = vars2[var][:]
            if not np.array_equal(data1, data2):
                print(f" - Variable {var} data differs")
                print(f"   - File 1 shape: {data1.shape}")
                print(f"   - File 2 shape: {data2.shape}")
                if data1.shape == data2.shape:
                    diff = np.abs(data1 - data2)
                    max_diff = np.max(diff)
                    print(f"   - Max difference: {max_diff}")

# Example usage
file1_path = 'l1r_11_updated_07032024/awe_l1r_q20_2023326T0108_00070_v01.nc'
file2_path = 'nc_files_with_mlcloud/awe_l1c_q20_2023326T0108_00070_v01.nc'
compare_netcdf_files(file1_path, file2_path)
