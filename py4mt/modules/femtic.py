import numpy as np

def read_femtic_data(data_file=None, site_file=None, data_type="z", out=True):

    data = []
    with open(data_file, "r") as f:
        iline = -1
        for line in f:
            iline = iline+1
            if iline > 0:
                l = line.split()
                l = [float(x) for x in l]
                # print(l)
                l[0] = int(l[0])
                data.append(l)
    data = np.array(data)

    info = []
    with open(site_file, "r") as f:
        for line in f:
            l = line.split(',')
            l[1] = float(l[1])
            l[2] = float(l[2])
            l[3] = int(l[3])
            print(l)
            info.append(l)
    info = np.array(info)


    """
     Site      Frequency     AppRxxCal       PhsxxCal      AppRxyCal       PhsxyCal
                             AppRyxCal       PhsyxCal      AppRyyCal       PhsyyCal
                             AppRxxObs       PhsxxObs      AppRxyObs       PhsxyObs
                             AppRyxObs       PhsyxObs      AppRyyObs       PhsyyObs
                             AppRxxErr       PhsxxErr      AppRxyErr       PhsxyErr
                             AppRyxErr       PhsyxErr      AppRyyErr       PhsyyErr
    """


    if "rhophas" in data_type.lower():
        data_dict =([
            ("cal_rhoxx", data[:, 2]),
            ("cal_rhoxy", data[:, 4]),
            ("cal_rhoyx", data[:, 6]),
            ("cal_rhoyy", data[:, 8]),

            ("cal_phsxx", data[:, 3]),
            ("cal_phsxy", data[:, 5]),
            ("cal_phsyx", data[:, 7]),
            ("cal_phsyy", data[:, 9]),


            ("obs_rhoxx", data[:, 10]),
            ("obs_rhoxy", data[:, 12]),
            ("obs_rhoyx", data[:, 14]),
            ("obs_rhoyy", data[:, 16]),

            ("obs_phsxx", data[:, 11]),
            ("obs_phsxy", data[:, 13]),
            ("obs_phsyx", data[:, 15]),
            ("obs_phsyy", data[:, 17]),

            ("obs_rhoxx_err", data[:, 18]),
            ("obs_rhoxy_err", data[:, 20]),
            ("obs_rhoyx_err", data[:, 22]),
            ("obs_rhoyy_err", data[:, 24]),

            ("obs_phsxx_err", data[:, 19]),
            ("obs_phsxy_err", data[:, 21]),
            ("obs_phsyx_err", data[:, 23]),
            ("obs_phsyy_err", data[:, 25]),


            ("per", data[:,1]),
            ("num", data[:,0]),



            ("lat", info[:,1]),
            ("lon", info[:,2]),
            ("pos", info[:,3]),
            ("sit", info[:,0][data[:,0].astype("int")-1]),
            ])

    return data_dict
