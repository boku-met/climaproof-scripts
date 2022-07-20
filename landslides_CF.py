#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:04:42 2022

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
    # requires precipitation data.
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

def moser_threshold(h):
    mth = (h ** -0.77) * h * 41.66
    return mth
        
def main():
    
    (path_in, path_out, datype) = user_data()
    
    if path_in.endswith("/"):
        None
    else:
        path_in += "/"
    infiles_pr = sorted(glob.glob(path_in+"pr_*.nc"))
          
    for file in infiles_pr:
        ds_in_pr = xr.open_dataset(file)
        chunkd = chunking_dict(file, ds_in_pr)
        if chunkd:
            ds_in_pr = ds_in_pr.chunk(chunkd)
        mask = xr.where(ds_in_pr.pr.isel(time=slice(0,60)).mean(dim="time", 
                                                                   skipna=True) 
                        >= -990, 1, np.nan).compute()
        print("*** Loading dataset {0} complete. Mask created.".format(file))
        
        # Calculate indicator with parallel processing
        mth24 = moser_threshold(24)
        mth48 = moser_threshold(48)
        mth72 = moser_threshold(72)
        
        pr_2day = ds_in_pr.pr.rolling(time=2, min_periods=2, center=False).sum().compute()
        pr_3day = ds_in_pr.pr.rolling(time=3, min_periods=3, center=False).sum().compute()
        
        cond_1d = xr.where(ds_in_pr.pr > mth24, 1, 0)
        cond_2d = xr.where(pr_2day > mth48, 1, 0)
        cond_3d = xr.where(pr_3day > mth72, 1, 0)
        
        cond_sum = (cond_1d + cond_2d + cond_3d).compute()
        cond_tot = xr.where(cond_sum >=1, 1, 0).compute()
        
        moser_exceed = cond_tot.resample(time = "A", skipna=True).sum().compute()
        moser_exceed = (moser_exceed * mask).compute()
                        
        print("--> Calculation of indicators for dataset {0} complete".format(file))
                
        # Add CF-conformal metadata
        
        # Attributes for the indicator variables:
        attr_dict = {"cell_methods":"time: sum within days time: sum over days "
                     "(days exceeding either 24h, 48h or 72h Moser-Hohensinn thresholds)", 
                     "coordinates": "time lat lon", 
                     "grid_mapping": "crs", "long_name": "Annual number of days exceeding "
                     "thresholds for either 1, 2 or 3-day precipitation sums. The thresholds "
                     "are calculated after the Moser-Hohensinn approach (Moser and Hohensinn 1983)", 
                     "standard_name": "number_of_days_with_precipitation_above_thresholds", 
                        "units": "1"}
            
        moser_exceed.attrs = attr_dict
        
        # Workaround for special calendars:
        try:
            moser_exceed.coords["time"] = ds_in_pr.time[ds_in_pr.time.dt.is_year_end]
        except AttributeError:
            time_resampled = ds_in_pr.time.resample(time="A")
            start_inds = np.array([x.start for x in time_resampled.groups.values()])
            end_inds = np.array([x.stop for x in time_resampled.groups.values()])
            end_inds[-1] = ds_in_pr.time.size
            end_inds -= 1
            start_inds = start_inds.astype(np.int32)
            end_inds = end_inds.astype(np.int32)
            
            moser_exceed.coords["time"] = ds_in_pr.time[end_inds]
                                                
        moser_exceed.time.attrs.update({"climatology":"climatology_bounds"})
        
        # Encoding and compression
        encoding_dict = {"_FillValue":-32767, "dtype":np.int16, 'zlib': True,
                         'shuffle': True,'complevel': 5, 'fletcher32': False, 
                         'contiguous': False}
        
        moser_exceed.encoding = encoding_dict
                                
        # Climatology variable
        climatology_attrs = {'long_name': 'time bounds', 'standard_name': 'time'}
        # Workaround for special calendars:
        try:
            climatology = xr.DataArray(np.stack((ds_in_pr.time[ds_in_pr.time.dt.is_year_start],
                                                 ds_in_pr.time[ds_in_pr.time.dt.is_year_end]), 
                                                axis=1), 
                                       coords={"time": moser_exceed.time, 
                                               "nv": np.arange(2, dtype=np.int16)},
                                       dims = ["time","nv"], 
                                       attrs=climatology_attrs)
        except AttributeError:
            climatology = xr.DataArray(np.stack((ds_in_pr.time[start_inds],
                                                 ds_in_pr.time[end_inds]), 
                                                axis=1), 
                                       coords={"time": moser_exceed.time, 
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
            
        file_attrs = {'title': 'Landslides',
         'institution': 'Institute of Meteorology and Climatology, University of '
         'Natural Resources and Life Sciences, Vienna, Austria',
         'source': modelname,
         'references': 'https://github.com/boku-met/climaproof-docs',
         'comment': 'Annual sum of days exceeding either 1-, 2-, or 3-day precipitation '
         'sum thresholds calculated after Moser and Hohensinn (1983). DOI: '
         'https://doi.org/10.1016/0013-7952(83)90003-0',
         'Conventions': 'CF-1.8'}
        
        ds_out = xr.Dataset(data_vars={"landslides": moser_exceed,
                                       "threshold_1day": mth24,
                                       "threshold_2day": mth48,
                                       "threshold_3day": mth72,
                                       "climatology_bounds": climatology, 
                                       "crs": crs}, 
                            coords={"time":moser_exceed.time, "lat": ds_in_pr.lat,
                                    "lon":ds_in_pr.lon},
                            attrs=file_attrs)
        
        if path_out.endswith("/"):
            None
        else:
            path_out += "/"
        outf = path_out + file.split("/")[-1].replace("pr_","landslides_")
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
