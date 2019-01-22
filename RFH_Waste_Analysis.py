# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 21:26:18 2018

@author: marc

IMPORTANT: 
    Run this file in the same folder as the 'Data_as_csv.zip' file
"""

#%% Setup
import zipfile
import pandas as pd
import datetime as dt
import numpy as np
from time import time
import matplotlib.pyplot as plt
import math
from scipy import stats
import statistics


#%% ===========================================================================
    # Main
# =============================================================================
def main():
    
    global t_s                  
    global headers
    global z
    global f_names
    global f_len
    global files
    global d_list
    global df
    global doordraai
    global datum
    global size
    t_s = time()            # starting time
    headers = []            # column header names
    z = "Data_as_csv.zip"   # name of zip file
    f_names = []            # list of file names
    f_len = []              # list of file lengths
    files = {}              # dict of f_names and actual files
    d_list = []             # list of rows to be dropped

    
    #%% Extract data
    try:
        z_file = zipfile.ZipFile(z)
        print("+++ Opening: {} \n+++ Following files were found:".format(z_file))
        l = len(z) - 3
        for fn in z_file.namelist():
            s = fn[l:]      # s is the file name without folder name
            if len(s) != 0 : 
                f_names.append(s)
                files[s] = pd.read_csv(z_file.extract(fn))
                f_len.append(files[s].shape)
        print("\tNames       ", f_names)
        print("\tData entries", f_len)
        prt()
        z_file.close()
        
    except Exception as e:
        z_file.close()
        print("+++ Open failed:", e)
    
    #%% Combine csvs
    df = pd.DataFrame()     # Full dataset
    for file in files:
        df = pd.concat([df, files[file]], ignore_index=True)
    headers = df.columns.values
    doordraai = headers[12]
    datum = headers[2]
    print("\n+++ Header names are:\n", headers)
    
    files = None            # Reduce memory requirement 
    size = df.shape         # Size of dataset
    print("\n+++ Total size: ", size)
    
    #%% Remove zero entries and replace str with int for 'aantal stuks doordraai'
    print("+++ Removing '0' entries and converting numeric values to integer.") 
    df_clean = rm_zr_c_int(df)
    df = df_clean
    df_clean = None         # Reduce memory requirement 
    prt()
    
    #%% Convert date time column to datetime objects
    print("+++ Converting dates to date objects")
    df_dates = c_date(df)
    df = df_dates
    df_dates = None         # Reduce memory requirement 
    prt()
    
    #%% Generate new column with ln of doordraai
    print("+++ Calculating natural log of doordraai values")
    l = []
    for dd in df[doordraai]:
        l.append(math.log(dd))
    lndd = pd.DataFrame({"Natural log of (stuks) Doordraai":l})
    df = pd.concat([df, lndd], axis=1)
    lndd = None
    headers = df.columns.values
    lndoordraai = headers[-1]
    prt()
    
    
    #%% Analyse general annual waste flow
    
    # print amt of unique values for each relevant column
    print("\n+++ Relevant columns summarised")
    print("\t------------------------+-------+--------------------------------------------")
    print("\t{name: <25}{length: <8}{names}".format(name="Column name", length="Unique", names="Names"))
    print("\t{name: <25}{length: <8}".format(name="", length="entries"))
    print("\t------------------------+-------+--------------------------------------------")
    for col in range(len(headers)):
        prnt_h_uniques(col)
    print("\t------------------------+-------+--------------------------------------------")
    
    # get general statistics per location and year
    print("\n+++ Relevant doordraai statistics by location and year")
    l_y_dict = {}           # Location and Year dict
    years = [2016, 2017, 2018]
    for l in df[headers[0]].unique():
        pdf = df.loc[lambda df: df[headers[0]] == l]
        for y in years:
            ypdf = pdf[(df[headers[2]] > dt.datetime(year=(y-1),month=12,day=31)) & \
                       (df[headers[2]] < dt.datetime(year=(y+1),month=1,day=1))]
            if len(ypdf) > 0:
                l_y_dict[str(l)+" "+str(y)] = ypdf
            ypdf = None
        l_y_dict[str(l)] = pdf
        pdf = None
        
    for ly in l_y_dict:
        dd = l_y_dict[ly][doordraai]
        lndd = l_y_dict[ly][lndoordraai]
        
        statdd = get_stat(dd, 2)
        print("\t{} Doordraai".format(ly))
        for s in statdd:
            print("\t{}: {}".format(s, statdd[s]))
        
        print("")
        statlndd = get_stat(lndd, 2)
        print("\t{} Natural log of Doordraai".format(ly))
        for s in statlndd:
            print("\t{}: {}".format(s, statlndd[s]))
            
        hist_norm(list(lndd), ly, lndoordraai, norm=True)
    prt()

    # plot of waste flow over the year by location
    

    # Get plant data
    l_p_dict = {}           # All PLANT (not flower) data by location
    print("+++ Extracting plant vs flower data")
    for l in df[headers[0]].unique():
        pdf = df.loc[lambda df: df[headers[0]] == l]
        ppdf = pdf[pdf[headers[3]] == "Planten"]
        l_p_dict[str(l)] = ppdf
        pdf = None
        ppdf = None
        
    for lp in l_p_dict:
        dd = l_p_dict[lp][doordraai]
        lndd = l_p_dict[lp][lndoordraai]
        
        statdd = get_stat(dd, 2)
        print("\t{} Doordraai".format(lp))
        for s in statdd:
            print("\t{}: {}".format(s, statdd[s]))
        
        print("")
        statlndd = get_stat(lndd, 2)
        print("\t{} Natural log of Doordraai".format(lp))
        for s in statlndd:
            print("\t{}: {}".format(s, statlndd[s]))
        
        orig = len(l_y_dict[lp])
        p = statdd["count"]
        p_s = int(round(p/orig*100, 0))
        print("\n\tshare of 'Planten': {}%".format(p_s))
            
        hist_norm(list(lndd), lp, lndoordraai, norm=True)
    prt()
    
    # Get full plant dataframe
    ppdf = df[df[headers[3]] == "Planten"]
    
    # Largest medians was chosen as filter to avoid large outliers (avg) and 
    # constant low volume (sum) to find 'large' supplies
    top = 100
    ppdf = cat_med_sort(ppdf, headers[9], doordraai, t=top)
    
    o_d = occ_ch(ppdf) # Occurrence dictionary
    
    delta_times = []
    for key in o_d: 
        value = o_d[key] 
        for dtm in value:
            delta_times.append(dtm)

    dt_occ = {} # delta time occurence
    # count of time delta
    for dtm in delta_times:
        dt_occ[dtm] = dt_occ.get(dtm, 0) + 1
    k_occ = list(dt_occ.keys())
    k_occ.sort()
    
    # Generate ecdf
    ecdf = [] # Emperical Cumulative Distribution Function
    for t in k_occ:
        l = len(ecdf)
        if l != 0:
            add = dt_occ[t] + ecdf[l - 1]
        else:
            add = dt_occ[t]
        ecdf.append(add)
    # NOTE: ecdf gives how much time there is between batches of plants 
    # being wasted NOT the amount of plants!
    
    # Percentage of plant occurence not wasted again
    days = 7
    perc_n_w = 1 - (ecdf[days -1] / ecdf[-1])
    
    print("{}% of plant batches is not wasted again within 7 days".format(round(perc_n_w*100), 1))
    
    plt.figure(figsize=(9,6))
    plt.subplot(121) #1 high, 2 wide, location 1
    x1 = k_occ[0:8]
    y1 = ecdf[0:8]
    plt.plot(x1, y1, color="red")
    
    x2 = k_occ[7:]
    y2 = ecdf[7:]
    plt.xlabel("Days since last \nwaste of product")
    plt.ylabel("Amount")
    plt.title("Emperical Cumulative Distribution \nFunction (ECDF) of days \nsince last waste of product")
    plt.plot(x2, y2)
    
    # Show zoomed and partitioned part of ecdf
    plt.subplot(122) #1 high, 2 wide, location 2
    part = 21 # Partition (amount of days to show)
    plt.plot(x1, y1, color="red")
    
    x2 = k_occ[7:part]
    y2 = ecdf[7:part]
    plt.xlabel("Days since last \nwaste of product")
    plt.title("Zoomed ECDF of days since \nlast waste of product")
    plt.plot(x2, y2)
    
    # line to seven day height
    x = [0, 7, 7]
    h = ecdf[7] # height
    y = [h, h, 0]
    plt.plot(x, y, color="black", linewidth=0.5)
    
    plt.savefig("T100_ECDF.png")
    plt.show()    

    ppdf = None
    
    
#%% ===========================================================================
    # Functions
# =============================================================================

def prt(final=False):
    # Print 
    """ Print the runtime of the program,
    use final=True if it is the final run time request"""
    ct = time()
    rt = round(ct-t_s, 2)
    if final == False:
        print("\t+++ The current run time is {} seconds.".format(rt))
    else:
        print("+++ The final run time is {} seconds.".format(rt))
        
def rm_zr_c_int(df):
    # Remove Zero Convert Integer
    """ Remove '0' entries and make str->int in 'aantal stuks doordraai' column """
    
    def mk_int(l):
        """ Make the passed string an integer 
            Assumes a '*,*' format to be passed. """
        i = l.find(",")
        if i > -1:
            l = l[:i]+l[i+1:]
        return int(l)
    
    count = 0
    for l in df[doordraai]:
        if l != 0:
            l = mk_int(l)
            df.at[count, doordraai] = l
        else:
            d_list.append(count)
        count += 1
        
        
    if len(d_list) == 0:
        print("\tNo zeroes found, converted 'Aantal stuks doordraai' to int")
        df_ = df
        df = None 
    else:
        keep = set(range(df.shape[0])) - set(d_list)
        df_ = df.take(list(keep))
        df = None
        print("\tReduced size:{} (zeroes removed from '{}')".format(df_.shape[0], doordraai))
        global size
        size = df_.shape
    
    return df_   

def c_date(df):
    # Convert Date
    """ Convert date column to datetime object """
    
    count = 0
    for l in df[datum]:
        d = dt.datetime.strptime(l, "%d-%m-%Y")
        df.at[count, datum] = d
        count += 1
    df_ = df
    df = None
    
    return df_

def prnt_h_uniques(h):
    # Print Header Uniques
    """ Print the name and amount of unique entries in the column, 
    if less than 5 entries, print entries """
    nm = headers[h]
    l = len(df[headers[h]].unique())
    if l <= 5:
        nms = df[headers[h]].unique()
    else:
        nms = ""
    print("\t{name: <25}{length: <8}{names}".format(name=nm, length=l, names=nms))
    
def get_stat(c, r=0):
    # Get Statistics
    """ Generate useful statistics on given pandas dataframe column,
        optional rounding argument"""
    s = {}
    s["count"] = c.count()
    s["mean"] = round(c.mean(),r)
    s["median"] = round(c.median(),r)
    s["standard deviation"] = round(c.std(),r)
    s["min"] = round(c.min(),r)
    s["max"] = round(c.max(),r)
    
    if r==0:
        for stat in s:
            s[stat] = int(s[stat])
    
    return s
    
def hist_norm(data, label, xaxis, bins=15, norm=False):
    # Histogram Normalised
    """ Plot normalised histogram """
    plt.hist(data, density = True, bins = bins, label = label)
    #plt.semilogy()
    
    if norm:
        mu, std = stats.norm.fit(data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth = 2)
    
    
    
    plt.legend(loc = "upper left")
    plt.xlabel(xaxis)
    plt.show()
    
def line_pl(line, date, mean, median):
    # Line Plot
    """ Plot line graph with median and mean line """
    xmax = len(line)
    plt.plot(y=line, x=date,\
         color = 'black', linewidth = 1, alpha = 0.5, \
         label = 'reference')
    plt.hlines(y = mean, xmin = 1, xmax = xmax, \
           color = 'black', label = 'reference mean')
    plt.hlines(y = median, xmin = 1, xmax = xmax, \
           color = 'black', label = 'reference mean')
    plt.semilogy()
    plt.ylabel('Temperature anomaly relative to 1951-1980 (°C)')
    plt.xlabel('Time (number of months)')
    
    plt.legend(loc = 'upper left')
    
    plt.show()
    
def cat_med_sort(df, col1, col2, t="all"):
    # Category Median Sort
    """ Return a pandas DF sorted on the median of the values in the column.
    col1 is used as grouping variable for the median of col2. """
        
    d = {}
    l = np.array([])
    l_c = 0 # line count
    for line in df[col1].unique():
        pdf = df[df[col1] == line][col2]
        med = statistics.median(pdf) # faster than pdf.median()
        
        if med not in d.keys():
            d[med] = [line]
            l = np.append(l, med)
        else:
            d[med].append(line)
        l_c += 1
        
    
    if t == "all":
        t = l_c
            
    l = np.sort(l)
    l = l[::-1]
    
    lt = np.array([])          
    count = 0
    while len(lt) < t:
        if len(d[l[count]]) == 1:
            lt = np.append(lt, d[l[count]][0])
        else:
            for x in d[l[count]]:
                lt = np.append(lt, x)
        if len(lt) > t:
            # If list is longer than top
            lt = lt[0:(t -1)]
            break
        
        count += 1 
        
        # Protection to break if lt is not updated anymore
        if count >= l_c :
            break
            
    
    pdf = pd.DataFrame()
    for l in lt:
        pdf = pd.concat([pdf, df[df[col1] == l]], ignore_index=True)
    
    return pdf

def occ_ch(df):
    # occurence check
    """ Check how often specific plant type occurs after each other and 
    return a dict with times between. The keys are product-location pairs,
    the values are lists of time differences. """
    
    o_d = {} # Occurence dict
    r_d = {} # Repeat dict
    for i, r in df.iterrows():
        d = r[2] # Date
        p = r[9] # Product
        l = r[0] # Location
        pl = p + " " + l # Product Location pair
        
        if pl in o_d:
            pd = o_d[pl] # Previous date
            dtm = int((d - pd).days) # Delta time
            
            # add to OR make new entry in r_d
            if pl in r_d:
                r_d[pl].append(dtm)
            else:
                r_d[pl] = [dtm]
        
        # Store last occured date for pl
        o_d[pl] = d
    return(r_d)


#%% ===========================================================================
    # Call main
# =============================================================================
    
if __name__ == '__main__':
    main()
    
    prt(final=True)