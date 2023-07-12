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
    # Please specify the path to the folder containing the data. This indicator
    # requires data for tmin and tmax; please put all the data for both variables
    # in the same folder.
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
    infiles_tmax = sorted(glob.glob(path_in+"tasmax*.nc"))
    infiles_tmin = sorted(glob.glob(path_in+"tasmin*.nc"))
    
    modname_tmax = [x.split("/")[-1].replace("tasmax_","") for x in infiles_tmax]
    modname_tmin = [x.split("/")[-1].replace("tasmin_","") for x in infiles_tmin]
          
    for i, mn_tmax in enumerate(modname_tmax):
        for j, mn_tmin in enumerate(modname_tmin):
            if mn_tmax == mn_tmin:    
                ds_in_tmax = xr.open_dataset(infiles_tmax[i])
                ds_in_tmin = xr.open_dataset(infiles_tmin[j])
                assert(ds_in_tmax.sizes == ds_in_tmin.sizes)
                chunkd = chunking_dict(infiles_tmax[i], ds_in_tmax)
                if chunkd:
                    ds_in_tmax = ds_in_tmax.chunk(chunkd)
                    ds_in_tmin = ds_in_tmin.chunk(chunkd)
                check_endyear = (ds_in_tmin.time.dt.month == 12) & (ds_in_tmin.time.dt.day == 30)
                time_fullyear = ds_in_tmin.time[check_endyear]
                years = np.unique(time_fullyear.dt.year)
                ds_in_tmin = ds_in_tmin.sel(time=slice(str(min(years)), str(max(years))))
                ds_in_tmax = ds_in_tmax.sel(time=slice(str(min(years)), str(max(years))))
                mask = xr.where(ds_in_tmax.tasmax.isel(time=slice(0,60)).mean(dim="time", 
                                                                           skipna=True) 
                                >= -990, 1, np.nan).compute()
                print("*** Loading datasets {0},\n"
                      "{1} complete. Mask created.".format(infiles_tmax[i],
                                                           infiles_tmin[j]))
                
                # Calculate indicator with parallel processing
                tmin_cond = xr.where(ds_in_tmax.tasmax > 0.0, ds_in_tmin.tasmin, np.nan).compute()
                tmax_cond = xr.where(tmin_cond <= -2.2, 1, 0).compute()
                fr_th_days = tmax_cond.resample(time = "M", skipna=True).sum().compute()
                fr_th_days = (fr_th_days * mask).compute()
                                
                print("--> Calculation of indicators for dataset {0},\n"
                      "{1} complete".format(infiles_tmax[i],infiles_tmin[j]))
                
                # Add CF-conformal metadata
                
                # Attributes for the indicator variables:
                attr_dict = {"cell_methods":"time: minimum within days time: "
                             "maximum within days time: sum over days (days below "
                             "-2.2°C tmin and above 0°C tmax temperature thresholds)", 
                             "coordinates": "time lat lon", 
                             "grid_mapping": "crs", "long_name": "number of days with "
                             "maximum temperature above 0°C and minimum temperature "
                             "below -2.2°C", 
                             "standard_name": "number_of_days_with_air_temperature_above_and_below_thresholds", 
                                "units": "1"}
                    
                fr_th_days.attrs = attr_dict
               
                time_resampled = ds_in_tmax.time.resample(time="M")
                start_inds = np.array([x.start for x in time_resampled.groups.values()])
                end_inds = np.array([x.stop for x in time_resampled.groups.values()])
                end_inds[-1] = ds_in_tmax.time.size
                end_inds -= 1
                start_inds = start_inds.astype(np.int32)
                end_inds = end_inds.astype(np.int32)
                
                fr_th_days.coords["time"] = ds_in_tmax.time[end_inds]
                                                    
                fr_th_days.time.attrs.update({"climatology":"climatology_bounds"})
                                    
                # Encoding and compression
                encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                                 'complevel': 1, 'fletcher32': False, 
                                 'contiguous': False}
                
                fr_th_days.encoding = encoding_dict
                                
                # Climatology variable
                climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
                climatology = xr.DataArray(np.stack((ds_in_tmax.time[start_inds],
                                                        ds_in_tmax.time[end_inds]), 
                                                    axis=1), 
                                            coords={"time": fr_th_days.time, 
                                                    "nv": np.arange(2, dtype=np.int16)},
                                            dims = ["time","nv"], 
                                            attrs=climatology_attrs)
                    
                climatology.encoding.update({"dtype":np.float64,'units': ds_in_tmax.time.encoding['units'],
                                             'calendar': ds_in_tmax.time.encoding['calendar']})
                
                crs = xr.DataArray(np.nan, attrs=ds_in_tmax.crs.attrs)
                        
                # Attributes for file
                if "model" in datype:
                    modelname = mn_tmax.replace(".nc","")
                else:
                    modelname = ("CARPATCLIM as primary source and E-OBS (Version 16.0) "
                    "data (regridded with ESMF_RegridWeightGen) as secondary source")
                    
                file_attrs = {'title': 'Freeze-thaw-days',
                 'institution': 'Institute of Meteorology and Climatology, University of '
                 'Natural Resources and Life Sciences, Vienna, Austria',
                 'source': modelname,
                 'references': 'https://github.com/boku-met/climaproof-docs',
                 'comment': 'Monthly sum of days with tmin <= -2.2 °C and tmax'
                 ' > 0 °C',
                 'Conventions': 'CF-1.8'}
                
                ds_out = xr.Dataset(data_vars={"freeze_thaw_days": fr_th_days, 
                                               "climatology_bounds": climatology, 
                                               "crs": crs}, 
                                    coords={"time":fr_th_days.time, "lat": ds_in_tmax.lat,
                                            "lon":ds_in_tmax.lon},
                                    attrs=file_attrs)
                
                if path_out.endswith("/"):
                    None
                else:
                    path_out += "/"
                outf = path_out + infiles_tmax[i].split("/")[-1].replace("tasmax","freeze_thaw_days")
                if os.path.isfile(outf):
                    print("File {0} already exists. Removing...".format(outf))
                    os.remove(outf)
                
                # Write final file to disk
                ds_out.to_netcdf(outf, unlimited_dims="time")
                print("Writing file {0} completed!".format(outf))
                ds_in_tmax.close()
                ds_in_tmin.close()
                ds_out.close()
    print("Successfully processed all input files!")
main()