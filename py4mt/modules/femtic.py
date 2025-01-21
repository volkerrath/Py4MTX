import numpy as np


def get_femtic_data(data_file=None, site_file=None, data_type="rhophas", out=True):

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
            l[3] = float(l[3])
            l[4] = int(l[4])
            print(l)
            info.append(l)
    info = np.array(info)

    sites = np.unique(info[:, 0])

    if "rhophas" in data_type.lower():

        """
         Site      Frequency
         AppRxxCal   PhsxxCal   AppRxyCal   PhsxyCal   AppRyxCal  PhsyxCal  AppRyyCal   PhsyyCal
         AppRxxObs   PhsxxObs   AppRxyObs   PhsxyObs   AppRyxObs  PhsyxObs  AppRyyObs   PhsyyObs
         AppRxxErr   PhsxxErr   AppRxyErr   PhsxyErr   AppRyxErr  PhsyxErr  AppRyyErr   PhsyyErr
        """
        data_dict = dict([
            ("sites", sites),
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


            ("frq", data[:, 1]),
            ("per", 1./data[:1]),
            ("num", data[:, 0]),
            ("lat", info[:, 1]),
            ("lon", info[:, 2]),
            ("elv", info[:, 3]),
            ("sit", info[:, 0][data[:, 0].astype("int")-1]),
        ])

    elif "imp" in data_type.lower():

        """
         Site      Frequency
         ReZxxCal   ImZxxCal   ReZxyCal   ImZxyCal   ReZyxCal  ImZyxCal  ReZyyCal   ImZyyCal
         ReZxxObs   ImZxxObs   ReZxyObs   ImZxyObs   ReZyxObs  ImZyxObs  ReZyyObs   ImZyyObs
         ReZxxErr   ImZxxErr   ReZxyErr   ImZxyErr   ReZyxErr  ImZyxErr  ReZyyErr   ImZyyErr
        """

        data_dict = dict([
            ("sites", sites),
            ("cal_rezxx", data[:, 2]),
            ("cal_rezxy", data[:, 4]),
            ("cal_rezyx", data[:, 6]),
            ("cal_rezyy", data[:, 8]),

            ("cal_imzxx", data[:, 3]),
            ("cal_imzxy", data[:, 5]),
            ("cal_imzyx", data[:, 7]),
            ("cal_imzyy", data[:, 9]),


            ("obs_rezxx", data[:, 10]),
            ("obs_rezxy", data[:, 12]),
            ("obs_rezyx", data[:, 14]),
            ("obs_rezyy", data[:, 16]),

            ("obs_imzxx", data[:, 11]),
            ("obs_imzxy", data[:, 13]),
            ("obs_imzyx", data[:, 15]),
            ("obs_imzyy", data[:, 17]),

            ("obs_rezxx_err", data[:, 18]),
            ("obs_rezxy_err", data[:, 20]),
            ("obs_rezyx_err", data[:, 22]),
            ("obs_rezyy_err", data[:, 24]),

            ("obs_imzxx_err", data[:, 19]),
            ("obs_imzxy_err", data[:, 21]),
            ("obs_imzyx_err", data[:, 23]),
            ("obs_imzyy_err", data[:, 25]),

            ("frq", data[:, 1]),
            ("per", 1./data[:1]),
            ("lat", info[:, 1]),
            ("lon", info[:, 2]),
            ("elv", info[:, 3]),
            ("sit", info[:, 0][data[:, 0].astype("int")-1]),
        ])

    elif "vtf" in data_type.lower():
        """
        Site    Frequency
        ReTzxCal   ImTzxCal   ReTzyCal   ImTzyCal
        ReTzxOb    ImTzxObs   ReTzyObs   ImTzyObs
        ReTzxErr   ImTzxErr   ReTzyErr   ImTzyErr
        """
        data_dict = dict([
            ("sites", sites),
            ("cal_retzx", data[:, 2]),
            ("cal_retzy", data[:, 4]),
            ("cal_imtzx", data[:, 3]),
            ("cal_imtzy", data[:, 5]),

            ("obs_retzx", data[:, 6]),
            ("obs_retzy", data[:, 8]),
            ("obs_imtzx", data[:, 7]),
            ("obs_imtzy", data[:, 9]),

            ("obs_retzx_err", data[:, 10]),
            ("obs_retzy_err", data[:, 12]),
            ("obs_imtzx_err", data[:, 11]),
            ("obs_imtzy_err", data[:, 13]),


            ("frq", data[:, 1]),
            ("per", 1./data[:1]),
            ("lat", info[:, 1]),
            ("lon", info[:, 2]),
            ("elv", info[:, 3]),
            ("sit", info[:, 0][data[:, 0].astype("int")-1]),
        ])

    return data_dict
