import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from dateutil import parser

def main(): 
    st.title("Car Complaint Scanner")
    st.write('Visualize complaint volume over time.')
    with st.spinner('Fetching data.'):
        df = scrape_data()
    st.write(f"Found {df.shape[0]} Complaints.")
    st.sidebar.header('Options')
    min_year, max_year = st.sidebar.slider("Model Years", 1970, datetime.datetime.now().year+1,[2014,2021])
    min_complaint = st.sidebar.slider("Min Complaints", 10, 2000, 75)
    with st.spinner('Filtering'):
        df = df[(df['model_year'] >= min_year) & (df['model_year'] <= max_year)].copy()
        make_model_year_count = df.groupby(['make_model_year']).sum()[['count']].reset_index()
        good_make_model_years = make_model_year_count[make_model_year_count['count'] >= min_complaint].sort_values('count')['make_model_year'].unique()
        df = df[df['make_model_year'].isin(good_make_model_years)].copy()
    good_makes = df['MAKETXT'].unique()
    good_models = df['MODELTXT'].unique()
    makes = st.sidebar.multiselect('Select Makes', good_makes, good_makes)
    models = st.sidebar.multiselect('Select Models', good_models, good_models)
    # Filter the main df 
    with st.spinner('Filtering'):        
        df = df[df['MAKETXT'].isin(makes)].copy()
        df = df[df['MODELTXT'].isin(models)].copy()
    st.write(f"Filtered to {df.shape[0]} Complaints.")

    with st.spinner('Getting Complaint Velocities...'): 
        df = add_complaint_velocities(df)

    st.write('Top Complaint Velocities Over the last 6m')


    st.subheader('Raw Quantities')
    fig1, ax1 = plt.subplots()
    pie_frame = df.groupby('MAKETXT').sum()[['count']].reset_index()
    plt_labels = pie_frame['MAKETXT']
    plt_sizes = pie_frame['count']
    ax1.pie(plt_sizes, explode=None, labels=plt_labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write(fig1)
    
def add_complaint_velocities(df_):
    df = df_.sort_values('fail_epoch').copy().reset_index(drop=True)
    df['make_model_cum'] = np.zeros(df.shape[0])
    df['make_model_year_cum'] = np.zeros(df.shape[0])
    for K,row in df.iterrows():
        subf = df[(df.make_model==row.make_model)&(df.fail_epoch<=row.fail_epoch)]
        df.loc[K,'make_model_cum'] = subf.shape[0]
        df.loc[K,'make_model_rate'] = df.loc[K,'make_model_cum']/(1+(row.fail_epoch-subf.fail_epoch.min()).total_seconds())/86400./365.
        df.loc[K,'make_model_year_cum'] = subf[(subf.make_model_year==row.make_model_year)&(subf.fail_epoch<=row.fail_epoch)].shape[0]
        df.loc[K,'make_model_year_rate'] = df.loc[K,'make_model_year_cum']/(1+(row.fail_epoch-datetime.datetime(row.model_year-1,1,1).total_seconds()))/86400./365.
    return df 

def parse_garbage_year(s): 
    try: 
        i = int(float(str(s)))
        if (i in range(1970,datetime.datetime.now().year+2)):
            return i 
        else:
            return np.nan
    except: 
        return np.nan

def parse_garbage_datestr(s_):
    s = str(s_)
#     print('s0',s)
    if type(s_) != str: 
        if s_ != s_: 
            return np.nan
    try: 
        s = str(int(float(s))) # no delimiting symbols
#         print('s1',s)
        if len(s) == 8:
            if int(s[:4]) in range(1970,datetime.datetime.now().year+1): 
                if int(s[4:6]) in range(1,13) and int(s[6:]) in range(1,32):
                    return datetime.datetime(int(s[:4]),int(s[4:6]),int(s[6:]))
                elif int(s[4:6]) in range(1,32) and int(s[6:]) in range(1,13):
                    return datetime.datetime(int(s[:4]), int(s[6:]), int(s[4:6]))
                else: 
                    return np.nan
        elif len(s) == 6:
            if int('20'+s[:2]) in range(1970,datetime.datetime.now().year+1):
                if int(s[2:4]) in range(1,13) and int(s[4:]) in range(1,32): 
                    return datetime.datetime(int('20'+s[:2]), int(s[2:4]),int(s[4:]))
                else:
                    return np.nan
            else: 
                return np.nan
        elif (len(s) == 5):
            if int('20'+s[:2]) in range(1970,datetime.datetime.now().year+1):
                if int(s[2:3]) in range(1,13) and int(s[3:]) in range(1,32): 
                    return datetime.datetime(int('20'+s[:2]), int(s[2:3]),int(s[3:]))
                else: 
                    return np.nan
            else: 
                return np.nan
        else: 
            return np.nan
    except: 
        # Try to do it using dateutil.parser. 
        try: 
            return parser.parse(s)
        except: 
            return np.nan

def to_epoch(dt): 
    if (type(dt)==datetime.datetime):
        return int(dt.timestamp())
    else:
        return np.nan

@st.cache
def scrape_data():
    # Here's the column names 
    col_raw = """1 CMPLID CHAR(9) NHTSA'S INTERNAL UNIQUE SEQUENCE NUMBER.
2 ODINO CHAR(9) NHTSA'S INTERNAL REFERENCE NUMBER.
3 MFR_NAME CHAR(40) MANUFACTURER'S NAME
4 MAKETXT CHAR(25) VEHICLE/EQUIPMENT MAKE
5 MODELTXT CHAR(256) VEHICLE/EQUIPMENT MODEL
6 YEARTXT CHAR(4) MODEL YEAR, 9999 IF UNKNOWN or N/A
7 CRASH CHAR(1) WAS VEHICLE INVOLVED IN A CRASH, 'Y' OR 'N'
8 FAILDATE CHAR(8) DATE OF INCIDENT (YYYYMMDD)
9 FIRE CHAR(1) WAS VEHICLE INVOLVED IN A FIRE 'Y' OR 'N'
10 INJURED NUMBER(2) NUMBER OF PERSONS INJURED
11 DEATHS NUMBER(2) NUMBER OF FATALITIES
12 COMPDESC CHAR(128) SPECIFIC COMPONENT'S DESCRIPTION
13 CITY CHAR(30) CONSUMER'S CITY
14 STATE CHAR(2) CONSUMER'S STATE CODE
15 VIN CHAR(11) VEHICLE'S VIN#
16 DATEA CHAR(8) DATE ADDED TO FILE (YYYYMMDD)
17 LDATE CHAR(8) DATE COMPLAINT RECEIVED BY NHTSA (YYYYMMDD)
18 MILES NUMBER(7) VEHICLE MILEAGE AT FAILURE
19 OCCURENCES NUMBER(4) NUMBER OF OCCURRENCES
20 CDESCR CHAR(2048) DESCRIPTION OF THE COMPLAINT
21 CMPL_TYPE CHAR(4) SOURCE OF COMPLAINT CODE:
22 POLICE_RPT_YN CHAR(1) WAS INCIDENT REPORTED TO POLICE 'Y' OR 'N'
23 PURCH_DT CHAR(8) DATE PURCHASED (YYYYMMDD)
24 ORIG_OWNER_YN CHAR(1) WAS ORIGINAL OWNER 'Y' OR 'N'
25 ANTI_BRAKES_YN CHAR(1) ANTI-LOCK BRAKES 'Y' OR 'N'
26 CRUISE_CONT_YN CHAR(1) CRUISE CONTROL 'Y' OR 'N'
27 NUM_CYLS NUMBER(2) NUMBER OF CYLINDERS
28 DRIVE_TRAIN CHAR(4) DRIVE TRAIN TYPE [AWD,4WD,FWD,RWD]
29 FUEL_SYS CHAR(4) FUEL SYSTEM CODE:
30 FUEL_TYPE CHAR(4) FUEL TYPE CODE:
31 TRANS_TYPE CHAR(4) VEHICLE TRANSMISSION TYPE [AUTO, MAN]
32 VEH_SPEED NUMBER(3) VEHICLE SPEED
33 DOT CHAR(20) DEPARTMENT OF TRANSPORTATION TIRE IDENTIFIER
34 TIRE_SIZE CHAR(30) TIRE SIZE
35 LOC_OF_TIRE CHAR(4) LOCATION OF TIRE CODE:
36 TIRE_FAIL_TYPE CHAR(4) TYPE OF TIRE FAILURE CODE:
37 ORIG_EQUIP_YN CHAR(1) WAS PART ORIGINAL EQUIPMENT 'Y' OR 'N'
38 MANUF_DT CHAR(8) DATE OF MANUFACTURE (YYYYMMDD)
39 SEAT_TYPE CHAR(4) TYPE OF CHILD SEAT CODE:
40 RESTRAINT_TYPE CHAR(4) INSTALLATION SYSTEM CODE;
41 DEALER_NAME CHAR(40) DEALER'S NAME
42 DEALER_TEL CHAR(20) DEALER'S TELEPHONE NUMBER
43 DEALER_CITY CHAR(30) DEALER'S CITY
44 DEALER_STATE CHAR(2) DEALER'S STATE CODE
45 DEALER_ZIP CHAR(10) DEALER'S ZIPCODE
46 PROD_TYPE CHAR(4) PRODUCT TYPE CODE:
47 REPAIRED_YN CHAR(1) WAS DEFECTIVE TIRE REPAIRED 'Y' OR 'N'
48 MEDICAL_ATTN CHAR(1) WAS MEDICAL ATTENTION REQUIRED 'Y' OR 'N'
49 VEHICLES_TOWED_YN CHAR(1) WAS VEHICLE TOWED 'Y' OR 'N' """
    cols = [line.split(' ')[1] for line in col_raw.split('\n')]
    df = pd.read_csv('https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip', sep='\t' , header=None, names = cols, compression='zip', error_bad_lines=False)  
    df['count']=np.ones(df.shape[0])
    df['model_year'] = df['YEARTXT'].map(parse_garbage_year)
    df['fail_date'] = df['FAILDATE'].map(parse_garbage_datestr)
    df['make_model_year'] = df['MAKETXT'].astype(str) + ' ' + df['MODELTXT'].astype(str) + ' ' + df['YEARTXT'].astype(str)
    df['make_model'] = df['MAKETXT'].astype(str) + ' ' + df['MODELTXT'].astype(str)
    df['fail_epoch'] = df['fail_date'].map(to_epoch)
    df = df[~np.isnan(df.fail_epoch)].copy()
    return df

main()