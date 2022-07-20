#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:25:57 2022

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
    # Please specify the path to the folder containing the data
    path_to_data = "" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = ""
    
    #Please select option: 'observation' or 'model'
    type_of_data = "" 
    
    return path_to_data, output_path, type_of_data

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
    
    (path_in, path_out, datype) = user_data()
    if path_in.endswith("/"):
        None
    else:
        path_in += "/"
    infiles = sorted(glob.glob(path_in+"tasmax*.nc"))
    
    for file in infiles:
        ds_in = xr.open_dataset(file)
        chunkd = chunking_dict(file, ds_in)
        if chunkd:
            ds_in = ds_in.chunk(chunkd)
        mask = xr.where(ds_in.tasmax.isel(time=slice(0,60)).mean(dim="time", 
                                                                   skipna=True) 
                        >= -990, 1, np.nan).compute()
        print("*** Loading dataset {0} complete. Mask created.".format(file))
        
        # Calculate indicator with parallel processing
        heatdays_30 = xr.where(ds_in.tasmax >= 30.0, 1, 0).compute()
        heatdays_40 = xr.where(ds_in.tasmax >= 40.0, 1, 0).compute()
        hd30_month = heatdays_30.resample(time = "M", skipna=True).sum().compute()
        hd40_month = heatdays_40.resample(time = "M", skipna=True).sum().compute()
        hd30_month = (hd30_month * mask).compute()
        hd40_month = (hd40_month * mask).compute()
        
        print("--> Calculation of indicators for dataset {0} complete".format(file))
        
        # Add CF-conformal metadata
        
        # Attributes for the indicator variables:
        attr_dict = {"cell_methods":"time: maximum within days time: sum over days "
                     "(days over temperature threshold)", "coordinates": "time lat lon", 
                     "grid_mapping": "crs", "long_name": "number of days with "
                     "maximum temperature above threshold", "standard_name":
                         "number_of_days_with_air_temperature_above_threshold", 
                        "units": "1"}
            
        hd30_month.attrs = attr_dict
        hd40_month.attrs = attr_dict
        hd30_month.attrs["long_name"] = hd30_month.attrs["long_name"].replace("threshold", "30 degC")
        hd40_month.attrs["long_name"] = hd40_month.attrs["long_name"].replace("threshold", "40 degC")
        # Workaround for special calendars:
        try:
            hd30_month.coords["time"] = ds_in.time[ds_in.time.dt.is_month_end]
            hd40_month.coords["time"] = ds_in.time[ds_in.time.dt.is_month_end]
        except AttributeError:
            time_resampled = ds_in.time.resample(time="M")
            start_inds = np.array([x.start for x in time_resampled.groups.values()])
            end_inds = np.array([x.stop for x in time_resampled.groups.values()])
            end_inds[-1] = ds_in.time.size
            end_inds -= 1
            start_inds = start_inds.astype(np.int32)
            end_inds = end_inds.astype(np.int32)
            
            hd30_month.coords["time"] = ds_in.time[end_inds]
            hd40_month.coords["time"] = ds_in.time[end_inds]
                        
        hd30_month.time.attrs.update({"climatology":"climatology_bounds"})
        hd40_month.time.attrs.update({"climatology":"climatology_bounds"})
            
        # Encoding and compression
        encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                         'shuffle': True,'complevel': 5, 'fletcher32': False, 
                         'contiguous': False}
        
        hd30_month.encoding = encoding_dict
        hd40_month.encoding = encoding_dict
        
        # Climatology variable
        climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
        # Workaround for special calendars:
        try:
            climatology = xr.DataArray(np.stack((ds_in.time[ds_in.time.dt.is_month_start],
                                                 ds_in.time[ds_in.time.dt.is_month_end]), 
                                                axis=1), 
                                       coords={"time": hd30_month.time, 
                                               "nv": np.arange(2, dtype=np.int16)},
                                       dims = ["time","nv"], 
                                       attrs=climatology_attrs)
        except AttributeError:
            climatology = xr.DataArray(np.stack((ds_in.time[start_inds],
                                                 ds_in.time[end_inds]), 
                                                axis=1), 
                                       coords={"time": hd30_month.time, 
                                               "nv": np.arange(2, dtype=np.int16)},
                                       dims = ["time","nv"], 
                                       attrs=climatology_attrs)
            
        climatology.encoding.update({"dtype":np.float64,'units': ds_in.time.encoding['units'],
                                     'calendar': ds_in.time.encoding['calendar']})
        
        crs = xr.DataArray(np.nan, attrs=ds_in.crs.attrs)
                
        # Attributes for file
        if "model" in datype:
            modelname = file.split("/")[-1].replace(".nc","")
        else:
            modelname = ("CARPATCLIM as primary source and E-OBS (Version 16.0) "
            "data (regridded with ESMF_RegridWeightGen) as secondary source")
            
        file_attrs = {'title': 'Heat Days',
         'institution': 'Institute of Meteorology and Climatology, University of '
         'Natural Resources and Life Sciences, Vienna, Austria',
         'source': modelname,
         'references': 'https://github.com/boku-met/climaproof-docs',
         'comment': 'The file contains indicators for different thresholds of '
         'daily maximum air temperature',
         'Conventions': 'CF-1.8'}
        
        ds_out = xr.Dataset(data_vars={"heatdays_30degC": hd30_month, 
                                       "heatdays_40degC": hd40_month, 
                                       "climatology_bounds": climatology, 
                                       "crs": crs}, 
                            coords={"time":hd30_month.time, "lat": ds_in.lat,
                                    "lon":ds_in.lon},
                            attrs=file_attrs)
        
        if path_out.endswith("/"):
            None
        else:
            path_out += "/"
        outf = path_out + file.split("/")[-1].replace("tasmax","heatdays")
        if os.path.isfile(outf):
            print("File {0} already exists. Removing...".format(outf))
            os.remove(outf)
        
        # Write final file to disk
        ds_out.to_netcdf(outf, unlimited_dims="time")
        print("Writing file {0} completed!".format(outf))
        ds_in.close()
        ds_out.close()
    print("Successfully processed all input files!")
main()
