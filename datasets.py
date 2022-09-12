import os
import pandas as pd
import urllib
import numpy as np
import pathlib
from io import BytesIO, StringIO
from zipfile import ZipFile
from scipy.io import arff


UCI_WEBSITE = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

data_url_dict = dict(
    yacht="00243/yacht_hydrodynamics.data",
    forest="forest-fires/forestfires.csv",
    airfoil="00291/airfoil_self_noise.dat",
    sml="00274/NEW-DATA.zip",
    parkinson="parkinsons/telemonitoring/parkinsons_updrs.data",
    power="00294/CCPP.zip",
    superconductivity="00464/superconduct.zip",
    protein="00265/CASP.csv",
    blog = "00304/BlogFeedback.zip",
    ctscan="00206/slice_localization_data.zip",
    virus = "00413/dataset.zip",
    gpu = "00440/sgemm_product_dataset.zip",
    airquality="00501/PRSA2017_Data_20130301-20170228.zip",
    road3d="00246/3D_spatial_network.txt",
    housepower="00235/household_power_consumption.zip"
)

columns_dict = dict(    
    yacht=[
        "Longitudinal position of the center of buoyancy",
        "Prismatic coefficient",
        "Length-displacement ratio",
        "Beam-draught ratio",
        "Length-beam ratio",
        "Froude number",
        "Residuary resistance per unit weight of displacement"],
    
    airfoil=[
        "Frequency",
        "Angle of attack",
        "Chord length",
        "Free-stream velocity",
        "Suction side displacement thickness",
        "Scaled sound pressure level",],
    
    road3d = [
    "OSM_ID: OpenStreetMap ID for each road segment or edge in the graph.",
    "LONGITUDE: Web Mercaptor (Google format) longitude",
    "LATITUDE: Web Mercaptor (Google format) latitude",
    "ALTITUDE: Height in meters.",],
)


target_dict = dict(
    sml = ["3:Temperature_Comedor_Sensor", "4:Temperature_Habitacion_Sensor"],
    parkinson = ['motor_UPDRS', 'total_UPDRS'],
    superconductivity = "critical_temp",
    protein = "RMSD",
    road3d = "ALTITUDE: Height in meters.",
    airquality = ["PM2.5","PM10","SO2","NO2","CO","O3"],
    housepower = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
    gpu = ["Run1 (ms)","Run2 (ms)","Run3 (ms)","Run4 (ms)"],
)


option_dict = dict(
    yacht = dict(delimiter=" ", header=None, usecols=[i for i in range(7)]),
    airfoil = dict(header=None, delimiter="\t"),
    road3d = dict(header=None),
    housepower = dict(delimiter=";", na_values=['?']),
)


def open_sml(path):
    resp = urllib.request.urlopen(path)
    zipfile = ZipFile(BytesIO(resp.read()))

    file1, file2 = zipfile.namelist()
    df1 = pd.read_csv(zipfile.open(file1), delimiter=" ")
    df2 = pd.read_csv(zipfile.open(file2), delimiter=" ")

    columns = df2.columns[2:]

    df = pd.concat((df1.drop(columns[-2:], axis=1),
                    df2.drop(columns[-2:], axis=1)),
                   ignore_index=True)
    df.columns = columns
    
    return df


def open_power(path):
    resp = urllib.request.urlopen(path)
    zipfile = ZipFile(BytesIO(resp.read()))

    df = pd.read_excel(zipfile.open('CCPP/Folds5x2_pp.xlsx'))
    return df


def open_superconductivity(path):
    resp = urllib.request.urlopen(path)
    zipfile = ZipFile(BytesIO(resp.read()))

    num_feat = pd.read_csv(zipfile.open('train.csv'))
    formula = pd.read_csv(zipfile.open('unique_m.csv'))
    formula = formula.drop(["critical_temp"], axis=1)
    
    df = pd.concat((num_feat, formula), axis=1)
    
    return df


def open_airquality(path):
    resp = urllib.request.urlopen(path)
    zipfile = ZipFile(BytesIO(resp.read()))
    
    df_list = []

    for file in zipfile.namelist():
        if ".csv" in file:
            df_list.append(pd.read_csv(zipfile.open(file)))

    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df


def open_gpu(path):
    resp = urllib.request.urlopen(path)
    zipfile = ZipFile(BytesIO(resp.read()))

    df = pd.read_csv(zipfile.open('sgemm_product.csv'))
    return df


def open_virus(path):
    resp = urllib.request.urlopen(path)
    zipfile = ZipFile(BytesIO(resp.read()))

    arr = []
    y = []

    for file in zipfile.namelist():
        if ".txt" in file:
            for line in zipfile.open(file).readlines():
                splits = line.decode("utf-8").split(" ")
                y.append(float(splits[0]))
                values = [int(x.split(":")[1]) for x in splits[1:] if ":" in x]
                columns = [int(x.split(":")[0]) for x in splits[1:] if ":" in x]
                new_line = np.zeros(482, dtype=int)
                new_line[columns] = values
                arr.append(new_line)
    arr = np.stack(arr, axis=0)
    y = np.array(y).reshape(-1, 1)
    arr = np.concatenate((arr, y), axis=1)

    df = pd.DataFrame(arr, columns=["%i"%i for i in range(482)]+["target"])
    return df


def open_blog(path):
    resp = urllib.request.urlopen(path)
    zipfile = ZipFile(BytesIO(resp.read()))

    df = pd.read_csv(zipfile.open('blogData_train.csv'), header=None)
    return df

custom_opening_dict = dict(
    sml = open_sml,
    power = open_power,
    superconductivity = open_superconductivity,
    airquality = open_airquality,
    virus = open_virus,
    blog = open_blog,
    gpu = open_gpu
)


def open_uci_dataset(dataset, online=True, path=None):
    if not dataset in data_url_dict:
        raise ValueError("Dataset `%s` is not available. Available datasets are: "
                         "%s"%(dataset, str(list(data_url_dict.keys()))))
    
    if online:
        path = UCI_WEBSITE + data_url_dict[dataset]
    
    elif path is None:
        dirname = os.path.dirname(__file__)
        if (not os.path.isdir(os.path.join(dirname, "datasets")) or
            not os.path.isdir(os.path.join(dirname, "datasets/uci_datasets"))):
            raise ValueError("No UCI datasets have been downloaded yet."
                             " Use argument `online=True` or download the dataset with"
                             " the function `datasets.download_uci(dataset)`.")
        
        list_files = os.listdir(os.path.join(dirname, "datasets/uci_datasets"))
        no_file_found = True
        for file in list_files:
            if dataset == file.split(".")[0]:
                no_file_found = False
                filename = file
        if no_file_found:
            raise ValueError("The dataset `%s` has not been downloaded yet."
                             " Use argument `online=True` or download the dataset with"
                             " the function `datasets.download_uci(dataset)`."%dataset)
        
        path = os.path.join(os.path.join(dirname, "datasets/uci_datasets"), filename)
        path = pathlib.Path(path).as_uri()
        
    else:
        path = pathlib.Path(path).as_uri()
        
    if dataset in custom_opening_dict:
        df = custom_opening_dict[dataset](path)
    else:
        if dataset in option_dict:
            kwargs = option_dict[dataset]
        else:
            kwargs = {}
        
        extension = data_url_dict[dataset].split(".")[-1]
        if extension in ["xls", "xlsx"]:
            df = pd.read_excel(path, **kwargs)
        else:
            df = pd.read_csv(path, **kwargs)
        
    if dataset in columns_dict:
        df.columns = columns_dict[dataset]    
    
    if dataset in target_dict:
        target = target_dict[dataset]
    else:
        target = df.columns[-1]
    
    if not isinstance(target, list):
        target = [target]
    
    y = df[target]
    X = df.drop(target, axis=1)
    
    return X, y


def download_uci(dataset):
    if not dataset in data_url_dict:
        raise ValueError("Dataset `%s` is not available. Available datasets are: "
                         "%s"%(dataset, str(list(data_url_dict.keys()))))
        
    print("Downloading...")
    dirname = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(dirname, "datasets")
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, "uci_datasets")
    if not os.path.isdir(path):
        os.mkdir(path)
    
    list_files = os.listdir(path)
    
    for file in list_files:
        if dataset == file.split(".")[0]:
            print("Dataset `%s` already downloaded."%dataset)
            return True
    
    url = UCI_WEBSITE + data_url_dict[dataset]
    filename = os.path.join(path, dataset+"."+url.split(".")[-1])
    urllib.request.urlretrieve(url, filename)
    
    print("Done! The dataset is stored at `%s`"%filename)
    return True