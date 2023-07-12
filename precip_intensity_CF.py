#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:33:23 2022

@author: benedikt.becsi<at>boku.ac.at
"""

import os
import glob
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
        
        min_year = min(years)
        max_year = max(years)
        if max_year - min_year < 29:
            print("Dataset {0} needs at least 30 years of data. Exiting.".format(file))
            exit()
        startyrs = np.arange(min_year, max_year-28)
        endyrs = np.arange(min_year+29, max_year+1)
        annual_ind = np.logical_and(ds_in_pr.time.dt.month == 7, ds_in_pr.time.dt.day == 15)
        pr_meta = xr.zeros_like(ds_in_pr.pr[annual_ind,:,:])
        pr_meta = pr_meta.sel(time=slice(str(endyrs.min()),str(endyrs.max())))
                
        # Calculate indicator with parallel processing
        def parallel_loop(sy, ey):
            pr_cur = ds_in_pr.pr.sel(time=slice(str(sy),str(ey)))
            pr_quant = pr_cur.quantile(0.999, dim="time", interpolation="linear", skipna=True).compute()
            return pr_quant
            
        parallel_results = Parallel(n_jobs=njobs)(delayed(parallel_loop)
                                                    (sy, ey) for sy, ey in 
                                                    zip(startyrs, endyrs))
        for i in range(pr_meta.time.size):
            pr_meta[i,:,:] = parallel_results[i]
          
        pr_meta = (pr_meta * mask).compute()
                        
        print("--> Calculation of indicators for dataset {0} complete".format(file))
                
        # Add CF-conformal metadata
        
        # Attributes for the indicator variables:
        attr_dict = {"cell_methods":"time: sum within days time: point over days "
                     "(99.9 percentile of daily precipitation sums within 30-year periods)", 
                     "coordinates": "time lat lon", 
                     "grid_mapping": "crs", "long_name": "99.9 percentile of daily "
                     "precipitation sums within 30-year periods", 
                     "standard_name": "precipitation_amount", 
                        "units": "kg m-2"}
            
        pr_meta.attrs = attr_dict
        
        time_re = ds_in_pr.time.resample(time="A")
        start_inds = np.array([x.start for x in time_re.groups.values()])
        start_inds = start_inds[:pr_meta.time.size]
        end_inds = np.array([x.stop for x in time_re.groups.values()])
        end_inds[-1] = ds_in_pr.time.size
        end_inds = end_inds[-pr_meta.time.size:]
        end_inds -= 1
        start_inds = start_inds.astype(np.int32)
        end_inds = end_inds.astype(np.int32)
        
        pr_meta.coords["time"] = ds_in_pr.time[end_inds]
                                                                
        pr_meta.time.attrs.update({"climatology":"climatology_bounds"})
        
        # Encoding and compression
        encoding_dict = {"_FillValue":9.96921e+36, "dtype":np.float32, 'zlib': True,
                         'complevel': 1, 'fletcher32': False, 
                         'contiguous': False}
        
        pr_meta.encoding = encoding_dict
                                
        # Climatology variable
        climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
                
        climatology = xr.DataArray(np.stack((ds_in_pr.time[start_inds],
                                             ds_in_pr.time[end_inds]), 
                                                axis=1), 
                                       coords={"time": pr_meta.time, 
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
            
        file_attrs = {'title': 'Precipitation Intensity',
         'institution': 'Institute of Meteorology and Climatology, University of '
         'Natural Resources and Life Sciences, Vienna, Austria',
         'source': modelname,
         'references': 'https://github.com/boku-met/climaproof-docs',
         'comment': '99.9 percentile of daily precipitation sums over 30-year '
         'periods',
         'Conventions': 'CF-1.8'}
        
        ds_out = xr.Dataset(data_vars={"precipitation_intensity": pr_meta,
                                       "climatology_bounds": climatology, 
                                       "crs": crs}, 
                            coords={"time":pr_meta.time, "lat": ds_in_pr.lat,
                                    "lon":ds_in_pr.lon},
                            attrs=file_attrs)
        
        if path_out.endswith("/"):
            None
        else:
            path_out += "/"
        outf = path_out + file.split("/")[-1].replace("pr_","precipitation_intensity_")
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