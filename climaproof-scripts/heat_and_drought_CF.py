#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:33:23 2022

@author: benedikt.becsi<at>boku.ac.at
"""

import os
import glob
import copy
from joblib import Parallel, delayed
import numpy as np
import xarray as xr

try: 
    os.nice(8-os.nice(0)) # set current nice level to 8, if it is lower 
except: # nice level already above 8
    pass

def user_data():
    # Please specify the path to the folder containing the data. This indicator
    # requires precipitation and tmax data; please put all the data 
    # for all variables in the same folder.
    path_to_data = "" 
    
    # Please specify the path to the folder where the output should be saved to
    output_path = ""
    
    # Please select option: 'observation' or 'model'
    type_of_data = ""
    
    # Please specify the number of available cores for parallel processing. 
    # This depends on the number of CPU cores / threads that can be used on 
    # your system. If unsure, leave "4" as default value.
    parallel_jobs = 4
                  
    return path_to_data, output_path, type_of_data, parallel_jobs

def consecutive_3d (array, condition, period_length, lt = False):
    """calculate consecutive events occuring in a 3D-array. 
    Returns a binary array with all events that occur equal to 
    or more often than [period length] times consecutively set to 1, 
    a 2D field with the maximum length of consecutive occurences (if they
    happen more often than [period length] times), and a 2D field of 0 and 1
    indicating whether [period length] consecutive events have occured on this
    location."""
    counter = np.zeros(array.shape, dtype=np.int32)
    consecutive_binary = np.zeros_like(counter)
    maxlen = np.zeros_like(counter)
    ntim = array.shape[0]
    counter[0,:,:] = (np.where(array[0,:,:] < condition, 1, 0) if lt == True
                      else np.where(array[0,:,:] > condition, 1, 0))
    t = 1
    print("starting counting...")
    while t <= ntim-1:
        pre_counter = counter[t-1,:,:]
        cur_counter = (np.where(array[t,:,:] < condition, 1, 0).astype("int32") if lt == True
                      else np.where(array[t,:,:] > condition, 1, 0).astype("int32"))
        cur_counter = pre_counter + cur_counter
        counter[t,:,:] = np.where(cur_counter > pre_counter, cur_counter, 0)
        t += 1
    print("counting finished. Creating indices...")
    consecutive_binary = np.where(counter > period_length, 1, 0)
    counter_ind_temp = (counter == period_length).nonzero()
    counter_ind = np.array((counter_ind_temp[0], counter_ind_temp[1], 
                            counter_ind_temp[2]), dtype=np.int32)
    del(counter_ind_temp)
    print("Index creation finished. Looping over all events...")
    for k in range(counter_ind.shape[1]):
        kstart = counter_ind[0,k] - (period_length-1)
        kend = counter_ind[0,k]+1
        consecutive_binary[kstart:kend, counter_ind[1,k], counter_ind[2,k]] = 1
    print("Looping over all events finished. Creating return data...")
    maxlen_t = np.where(counter >= period_length, counter, 0)
    maxlen = np.max(maxlen_t, axis=0)
    nevent = np.sum(consecutive_binary, axis=0)
    #binary = np.where(nevent > 0,1,0).astype(np.int32)
    print("calculating consecutive events...Finished!")
    return (consecutive_binary, maxlen, nevent)

def main():
    (path_in, path_out, datype, njobs) = user_data()
    
    if path_in.endswith("/"):
        None
    else:
        path_in += "/"
    infiles_pr = sorted(glob.glob(path_in+"pr_*.nc"))
    infiles_tmax = sorted(glob.glob(path_in+"tasmax_*.nc"))
    
    modname_tmax = [x.split("/")[-1].replace("tasmax_","") for x in infiles_tmax]
    modname_pr = [x.split("/")[-1].replace("pr_","") for x in infiles_pr]      
    
    for j, mn_tmax in enumerate(modname_tmax):
        for k, mn_pr in enumerate(modname_pr):
            if mn_tmax == mn_pr:                   
                ds_in_tmax = xr.open_dataset(infiles_tmax[j])
                ds_in_pr = xr.open_dataset(infiles_pr[k])
                assert(ds_in_tmax.sizes == ds_in_pr.sizes)
                
                mask = xr.where(ds_in_pr.pr.isel(time=slice(0,60)).mean(dim="time", 
                                                                           skipna=True) 
                                >= -990, 1, np.nan).compute()
                print("*** Loading datasets {0},\n"
                      "{1} complete. Mask created.".format(infiles_tmax[j],
                                                           infiles_pr[k]))
                
                annual_ind = np.logical_and(ds_in_pr.time.dt.month == 7, ds_in_pr.time.dt.day == 15)
                cdd5 = xr.zeros_like(ds_in_pr.pr[annual_ind,:,:])
                cdd7 = copy.deepcopy(cdd5)
                years = np.unique(ds_in_pr.time.dt.year)
                # Calculate indicator with parallel processing
                def parallel_loop(year):
                    pr_cur = ds_in_pr.pr.sel(time=str(year)).values
                    tmax_cur = ds_in_tmax.tasmax.sel(time=str(year)).values
                    cdd5, mxlen5, nevent5 = consecutive_3d(pr_cur, 1.0, 5, lt=True)
                    cdd7, mxlen7, nevent7 = consecutive_3d(pr_cur, 1.0, 7, lt=True)
                    dry_mask_5 = np.where(cdd5 == 1, tmax_cur, 0)
                    dry_mask_7 = np.where(cdd7 == 1, tmax_cur, 0)
                    dry_mask_5 = np.nansum(np.where(dry_mask_5 >= 30, 1, 0), axis = 0)
                    dry_mask_7 = np.nansum(np.where(dry_mask_7 >= 30, 1, 0), axis = 0)
                    return dry_mask_5, dry_mask_7, year
                
                parallel_results = Parallel(n_jobs=njobs)(delayed(parallel_loop)(year) for year in years)
                
                cdd5.values = [x[0] for x in parallel_results]
                cdd7.values = [x[1] for x in parallel_results]
                          
                cdd5 = (cdd5 * mask).compute()
                cdd7 = (cdd7 * mask).compute()
                print("--> Calculation of indicators for dataset {0},\n"
                      "{1} complete".format(infiles_tmax[j], infiles_pr[k]))
                        
                # Add CF-conformal metadata
        
                # Attributes for the indicator variables
                attr_dict = {"cell_methods":"time: sum within days time: sum over days "
                             "(xx)", 
                             "coordinates": "time lat lon", 
                             "grid_mapping": "crs", "long_name": "yy", 
                             "standard_name": "zz", 
                                "units": "1"}
                    
                cdd5.attrs, cdd7.attrs = (attr_dict, attr_dict)
                cdd5.attrs.update({"cell_methods":cdd5.attrs["cell_methods"].replace("xx",
                                "days in consecutive 5-day dry periods with precipitation < 1mm "
                                "and maximum temperature >= 30°C"),
                                "long_name":cdd5.attrs["long_name"].replace("yy",
                                "annual number of hot and dry days (pr < 1 mm, tmax >= 30°C) "
                                "within consecutive 5-day dry periods"),
                                "standard_name": cdd5.attrs["standard_name"].replace("zz",
                                "number_of_days_with_precipitation_amount_below_threshold")})
                cdd7.attrs.update({"cell_methods":cdd7.attrs["cell_methods"].replace("xx",
                                "days in consecutive 7-day dry periods with precipitation < 1mm "
                                "and maximum temperature >= 30°C"),
                                "long_name":cdd7.attrs["long_name"].replace("yy",
                                "annual number of hot and dry days (pr < 1 mm, tmax >= 30°C) "
                                "within consecutive 7-day dry periods"),
                                "standard_name": cdd7.attrs["standard_name"].replace("zz",
                                "number_of_days_with_precipitation_amount_below_threshold")})
                
                try:
                    cdd5.coords["time"] = ds_in_pr.time[ds_in_pr.time.dt.is_year_end]
                    cdd7.coords["time"] = ds_in_pr.time[ds_in_pr.time.dt.is_year_end]
                    
                except AttributeError:
                    time_resampled = ds_in_pr.time.resample(time="A")
                    start_inds = np.array([x.start for x in time_resampled.groups.values()])
                    end_inds = np.array([x.stop for x in time_resampled.groups.values()])
                    end_inds[-1] = ds_in_pr.time.size
                    end_inds -= 1
                    start_inds = start_inds.astype(np.int32)
                    end_inds = end_inds.astype(np.int32)
                    
                    cdd5.coords["time"] = ds_in_pr.time[end_inds]
                    cdd7.coords["time"] = ds_in_pr.time[end_inds]
                               
                cdd5.time.attrs.update({"climatology":"climatology_bounds"})
                cdd7.time.attrs.update({"climatology":"climatology_bounds"})
                        
                # Encoding and compression
                encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                                 'shuffle': True,'complevel': 5, 'fletcher32': False, 
                                 'contiguous': False}
                
                cdd5.encoding = encoding_dict
                cdd7.encoding = encoding_dict
                                                
                # Climatology variable
                climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
                        
                try:
                    climatology = xr.DataArray(np.stack((ds_in_pr.time[ds_in_pr.time.dt.is_year_start],
                                                         ds_in_pr.time[ds_in_pr.time.dt.is_year_end]), 
                                                        axis=1), 
                                               coords={"time": cdd5.time, 
                                                       "nv": np.arange(2, dtype=np.int16)},
                                               dims = ["time","nv"], 
                                               attrs=climatology_attrs)
                except AttributeError:
                    climatology = xr.DataArray(np.stack((ds_in_pr.time[start_inds],
                                                         ds_in_pr.time[end_inds]), 
                                                        axis=1), 
                                               coords={"time": cdd5.time, 
                                                       "nv": np.arange(2, dtype=np.int16)},
                                               dims = ["time","nv"], 
                                               attrs=climatology_attrs)
                            
                climatology.encoding.update({"dtype":np.float64,'units': ds_in_pr.time.encoding['units'],
                                             'calendar': ds_in_pr.time.encoding['calendar']})
                
                crs = xr.DataArray(np.nan, attrs=ds_in_pr.crs.attrs)
        
                # Attributes for file
                if "model" in datype:
                    modelname = mn_tmax.replace(".nc","")
                else:
                    modelname = ("CARPATCLIM as primary source and E-OBS (Version 16.0) "
                    "data (regridded with ESMF_RegridWeightGen) as secondary source")
                    
                file_attrs = {'title': 'Drought and Heat',
                 'institution': 'Institute of Meteorology and Climatology, University of '
                 'Natural Resources and Life Sciences, Vienna, Austria',
                 'source': modelname,
                 'references': 'https://github.com/boku-met/climaproof-docs',
                 'comment': 'This file contains annual number of days that are '
                 'within consecutive dry periods of different thresholds and '
                 'hava a maximum temperature of >= 30°C',
                 'Conventions': 'CF-1.8'}
                
                ds_out = xr.Dataset(data_vars={"drought_and_heat_5_day_periods": cdd5,
                                               "drought_and_heat_7_day_periods": cdd7,                                   
                                               "climatology_bounds": climatology, 
                                               "crs": crs}, 
                                    coords={"time":cdd5.time, "lat": ds_in_pr.lat,
                                            "lon":ds_in_pr.lon},
                                    attrs=file_attrs)
                
                if path_out.endswith("/"):
                    None
                else:
                    path_out += "/"
                outf = path_out + infiles_pr[k].split("/")[-1].replace("pr_","drought_and_heat_")
                if os.path.isfile(outf):
                    print("File {0} already exists. Removing...".format(outf))
                    os.remove(outf)
                
                # Write final file to disk
                ds_out.to_netcdf(outf, unlimited_dims="time")
                print("Writing file {0} completed!".format(outf))
                ds_in_pr.close()
                ds_in_tmax.close()
                ds_out.close()
    print("Successfully processed all input files!")
main()
