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
    # requires precipitation data.
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
          
    for file in infiles_pr:
        ds_in_pr = xr.open_dataset(file)
        check_endyear = (ds_in_pr.time.dt.month == 12) & (ds_in_pr.time.dt.day == 30)
        time_fullyear = ds_in_pr.time[check_endyear]
        years = np.unique(time_fullyear.dt.year)
        ds_in_pr = ds_in_pr.sel(time=slice(str(min(years)), str(max(years))))
        mask = xr.where(ds_in_pr.pr.isel(time=slice(0,60)).mean(dim="time", 
                                                                   skipna=True) 
                        >= -990, 1, np.nan).compute()
        print("*** Loading dataset {0} complete. Mask created.".format(file))
        annual_ind = np.logical_and(ds_in_pr.time.dt.month == 7, ds_in_pr.time.dt.day == 15)
        cdd5 = xr.zeros_like(ds_in_pr.pr[annual_ind,:,:])
        cdd7 = copy.deepcopy(cdd5)
        maxlen5 = copy.deepcopy(cdd5)
        maxlen7 = copy.deepcopy(cdd5)
        years = np.unique(ds_in_pr.time.dt.year)
        # Calculate indicator with parallel processing
        def parallel_loop(year):
            pr_cur = ds_in_pr.pr.sel(time=str(year))
            summer_ind = np.logical_and(pr_cur.time.dt.month >= 4, pr_cur.time.dt.month <= 9)
            pr_cur = pr_cur[summer_ind,:,:].values
            cdd5, mxlen5, nevent5 = consecutive_3d(pr_cur, 1.0, 5, lt=True)
            cdd7, mxlen7, nevent7 = consecutive_3d(pr_cur, 1.0, 7, lt=True)
            return nevent5, mxlen5, nevent7, mxlen7, year
        
        parallel_results = Parallel(n_jobs=njobs)(delayed(parallel_loop)(year) for year in years)
        
        cdd5.values = [x[0] for x in parallel_results]
        maxlen5.values = [x[1] for x in parallel_results]
        cdd7.values = [x[2] for x in parallel_results]
        maxlen7.values = [x[3] for x in parallel_results]
          
        cdd5 = (cdd5 * mask).compute()
        cdd7 = (cdd7 * mask).compute()
        maxlen5 = (maxlen5 * mask).compute()
        maxlen7 = (maxlen7 * mask).compute()                        
        print("--> Calculation of indicators for dataset {0} complete".format(file))
                
        # Add CF-conformal metadata
        
        # Attributes for the indicator variables:
        attr_dict = {"cell_methods":"time: sum within days time: sum over days "
                     "(xx)", 
                     "coordinates": "time lat lon", 
                     "grid_mapping": "crs", "long_name": "yy", 
                     "standard_name": "zz", 
                        "units": "1"}
            
        cdd5.attrs, cdd7.attrs, maxlen5.attrs, maxlen7.attrs = (attr_dict, 
        attr_dict, attr_dict, attr_dict)
        cdd5.attrs.update({"cell_methods":cdd5.attrs["cell_methods"].replace("xx",
                        "days in consecutive 5-day periods with precipitation < 1mm"),
                        "long_name":cdd5.attrs["long_name"].replace("yy",
                        "number of dry days (pr < 1 mm) within consecutive 5-day "
                        "periods during the summer half year (April-September)"),
                        "standard_name": cdd5.attrs["standard_name"].replace("zz",
                        "number_of_days_with_precipitation_amount_below_threshold")})
        cdd7.attrs.update({"cell_methods":cdd7.attrs["cell_methods"].replace("xx",
                        "days in consecutive 7-day periods with precipitation < 1mm"),
                        "long_name":cdd7.attrs["long_name"].replace("yy",
                        "number of dry days (pr < 1 mm) within consecutive 7-day "
                        "periods during the summer half year (April-September)"),
                        "standard_name": cdd7.attrs["standard_name"].replace("zz",
                        "number_of_days_with_precipitation_amount_below_threshold")})
        maxlen5.attrs.update({"cell_methods":maxlen5.attrs["cell_methods"].replace("xx",
                        "longest period of at least 5 consecutive dry days with precipitation < 1mm"),
                        "long_name":maxlen5.attrs["long_name"].replace("yy",
                        "longest consecutive period of at least 5 dry days "
                        "(pr < 1 mm) during the summer half year (April-September)"),
                        "standard_name": maxlen5.attrs["standard_name"].replace("zz",
                        "spell_length_of_days_with_precipitation_amount_below_threshold")})
        maxlen7.attrs.update({"cell_methods":maxlen7.attrs["cell_methods"].replace("xx",
                        "longest period of at least 7 consecutive dry days with precipitation < 1mm"),
                        "long_name":maxlen7.attrs["long_name"].replace("yy",
                        "longest consecutive period of at least 7 dry days "
                        "(pr < 1 mm) during the summer half year (April-September)"),
                        "standard_name": maxlen7.attrs["standard_name"].replace("zz",
                        "spell_length_of_days_with_precipitation_amount_below_threshold")})
        
        start_inds = np.logical_and(ds_in_pr.time.dt.month == 4, ds_in_pr.time.dt.day == 1)
        end_inds = np.logical_and(ds_in_pr.time.dt.month == 9, ds_in_pr.time.dt.day == 30)
        cdd5.coords["time"] = ds_in_pr.time[end_inds]
        cdd7.coords["time"] = ds_in_pr.time[end_inds]
        maxlen5.coords["time"] = ds_in_pr.time[end_inds]
        maxlen7.coords["time"] = ds_in_pr.time[end_inds]
        
        cdd5.time.attrs.update({"climatology":"climatology_bounds"})
        cdd7.time.attrs.update({"climatology":"climatology_bounds"})
        maxlen5.time.attrs.update({"climatology":"climatology_bounds"})
        maxlen7.time.attrs.update({"climatology":"climatology_bounds"})
        
        # Encoding and compression
        encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                         'complevel': 1, 'fletcher32': False, 
                         'contiguous': False}
        
        cdd5.encoding = encoding_dict
        cdd7.encoding = encoding_dict
        maxlen5.encoding = encoding_dict
        maxlen7.encoding = encoding_dict
                                
        # Climatology variable
        climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
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
            modelname = file.split("/")[-1].replace(".nc","")
        else:
            modelname = ("CARPATCLIM as primary source and E-OBS (Version 16.0) "
            "data (regridded with ESMF_RegridWeightGen) as secondary source")
            
        file_attrs = {'title': 'Concecutive Dry Days',
         'institution': 'Institute of Meteorology and Climatology, University of '
         'Natural Resources and Life Sciences, Vienna, Austria',
         'source': modelname,
         'references': 'https://github.com/boku-met/climaproof-docs',
         'comment': 'This file contains the number of days and maximum period '
         'length of consecutive dry days (daily precipitation sum < 1 mm) '
         'for different period lengths during the summer half-year (Apr-Sep)',
         'Conventions': 'CF-1.8'}
        
        ds_out = xr.Dataset(data_vars={"number_of_consecutive_dry_days_5_day_periods": cdd5,
                                       "number_of_consecutive_dry_days_7_day_periods": cdd7,
                                       "maximum_spell_lenght_5_day_periods": maxlen5,
                                       "maximum_spell_lenght_7_day_periods": maxlen7,                                       
                                       "climatology_bounds": climatology, 
                                       "crs": crs}, 
                            coords={"time":cdd5.time, "lat": ds_in_pr.lat,
                                    "lon":ds_in_pr.lon},
                            attrs=file_attrs)
        
        if path_out.endswith("/"):
            None
        else:
            path_out += "/"
        outf = path_out + file.split("/")[-1].replace("pr_","consecutive_dry_days_")
        if os.path.isfile(outf):
            print("File {0} already exists. Removing...".format(outf))
            os.remove(outf)
        
        # Write final file to disk
        ds_out.to_netcdf(outf, unlimited_dims="time")
        print("Writing file {0} completed!".format(outf))
        ds_in_pr.close()
        ds_out.close()
    print("Successfully processed all input files!")
main()