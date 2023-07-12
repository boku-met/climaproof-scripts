#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:45:55 2022

@author: benedikt.becsi<at>boku.ac.at
"""

import os
import glob
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

def user_data():
    # Please specify the path to the folder containing the data. This indicator
    # requires wind speed data, and sets the threshold according to reference
    # observational data. Please set the "reference_obs_file" variable directly
    # to the file with the reference observational data. It should contain the 
    # period 1981-2010. The reference file and the files you want to calculate
    # the indicator for need to have the same spatial resolution!
    path_to_data = "" 
    reference_obs_file = ""

    # Please specify the path to the folder where the output should be saved to
    output_path = ""

    #Please select option: 'observation' or 'model'
    type_of_data = "" 

    return path_to_data, reference_obs_file, output_path, type_of_data

def chunking_dict(filename, ds_in):
    chunkdict = None
    fsize = os.stat(filename).st_size / 1000000
    if fsize > 200:
        chunk_div = fsize / 100
        chunkdict = {"time":int(round(ds_in.time.size / chunk_div))}
        if chunkdict["time"] == 0:
            chunkdict["time"] = 1
    return chunkdict
        
def main():
    
    (path_in, infile_ref, path_out, datype) = user_data()
    
    if path_in.endswith("/"):
        None
    else:
        path_in += "/"
    infiles_wind = sorted(glob.glob(path_in+"sfcWind_*.nc"))
    
    ds_ref = xr.open_dataset(infile_ref)
    ds_ref = ds_ref.sel(time=slice("1981","2010"))
    ref_thresholds = ds_ref.sfcWind.quantile(0.999, dim="time", interpolation="linear", skipna=True).compute()
          
    for file in infiles_wind:
        ds_in_wind = xr.open_dataset(file)
        chunkd = chunking_dict(file, ds_in_wind)
        if chunkd:
            ds_in_wind = ds_in_wind.chunk(chunkd)
        check_endyear = (ds_in_wind.time.dt.month == 12) & (ds_in_wind.time.dt.day == 30)
        time_fullyear = ds_in_wind.time[check_endyear]
        years = np.unique(time_fullyear.dt.year)
        ds_in_wind = ds_in_wind.sel(time=slice(str(min(years)), str(max(years))))
        mask = xr.where(ds_in_wind.sfcWind.isel(time=slice(0,60)).mean(dim="time", 
                                                                   skipna=True) 
                        >= -990, 1, np.nan).compute()
        print("*** Loading dataset {0} complete. Mask created.".format(file))
        
        # Calculate indicator with parallel processing
        exceed_th = xr.where(ds_in_wind.sfcWind >= ref_thresholds, 1, 0).compute()
        extreme_wind = exceed_th.resample(time="A", skipna=True).sum().compute()
        extreme_wind = (extreme_wind * mask).compute()
        
        print("--> Calculation of indicators for dataset {0} complete".format(file))
                
        # Add CF-conformal metadata
        
        # Attributes for the indicator variables:
        attr_dict = {"cell_methods":"time: mean within days time: sum over days "
                     "(days exceeding the 99.9 percentile threshold of daily "
                     "wind speeds in the reference period 1981-2010)", 
                     "coordinates": "time lat lon", 
                     "grid_mapping": "crs", "long_name": "Annual number of days exceeding "
                     "99.9 percentile threshold of daily wind speeds taken from "
                     "observational data (1981-2010)", 
                     "standard_name": "number_of_days_with_wind_speed_above_threshold", 
                        "units": "1"}
            
        extreme_wind.attrs = attr_dict
        
        time_resampled = ds_in_wind.time.resample(time="A")
        start_inds = np.array([x.start for x in time_resampled.groups.values()])
        end_inds = np.array([x.stop for x in time_resampled.groups.values()])
        end_inds[-1] = ds_in_wind.time.size
        end_inds -= 1
        start_inds = start_inds.astype(np.int32)
        end_inds = end_inds.astype(np.int32)
            
        extreme_wind.coords["time"] = ds_in_wind.time[end_inds]
                                                
        extreme_wind.time.attrs.update({"climatology":"climatology_bounds"})
        
        # Encoding and compression
        encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                         'complevel': 1, 'fletcher32': False, 
                         'contiguous': False}
        
        extreme_wind.encoding = encoding_dict
                                
        # Climatology variable
        climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
        climatology = xr.DataArray(np.stack((ds_in_wind.time[start_inds],
                                                ds_in_wind.time[end_inds]), 
                                            axis=1), 
                                    coords={"time": extreme_wind.time, 
                                            "nv": np.arange(2, dtype=np.int16)},
                                    dims = ["time","nv"], 
                                    attrs=climatology_attrs)
            
        climatology.encoding.update({"dtype":np.float64,'units': ds_in_wind.time.encoding['units'],
                                     'calendar': ds_in_wind.time.encoding['calendar']})
        
        crs = xr.DataArray(np.nan, attrs=ds_in_wind.crs.attrs)
                
        # Attributes for file
        if "model" in datype:
            modelname = file.split("/")[-1].replace(".nc","")
        else:
            modelname = ("CARPATCLIM as primary source and E-OBS (Version 16.0) "
            "data (regridded with ESMF_RegridWeightGen) as secondary source")
            
        file_attrs = {'title': 'Extreme Wind',
         'institution': 'Institute of Meteorology and Climatology, University of '
         'Natural Resources and Life Sciences, Vienna, Austria',
         'source': modelname,
         'references': 'https://github.com/boku-met/climaproof-docs',
         'comment': 'Annual sum of days exceeding the 99.9 percentile of daily '
         'wind speeds in the reference period 1981-2010',
         'Conventions': 'CF-1.8'}
        
        ds_out = xr.Dataset(data_vars={"extreme_wind": extreme_wind,
                                       "reference_thresholds": ref_thresholds,
                                       "climatology_bounds": climatology, 
                                       "crs": crs}, 
                            coords={"time": extreme_wind.time, "lat": ds_in_wind.lat,
                                    "lon":ds_in_wind.lon},
                            attrs=file_attrs)
        
        if path_out.endswith("/"):
            None
        else:
            path_out += "/"
        outf = path_out + file.split("/")[-1].replace("sfcWind_","extreme_wind_")
        if os.path.isfile(outf):
            print("File {0} already exists. Removing...".format(outf))
            os.remove(outf)
        
        # Write final file to disk
        ds_out.to_netcdf(outf, unlimited_dims="time")
        print("Writing file {0} completed!".format(outf))
        ds_in_wind.close()
        ds_out.close()
    print("Successfully processed all input files!")
main()