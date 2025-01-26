import numpy as np
from sys import exit as error


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
            # print(l)
            info.append(l)
    info = np.array(info)

    sites = np.unique(data[:, 0]).astype("int")-1

    head_dict = dict([
        ("sites", sites),
        ("frq", data[:, 1]),
        ("per", 1./data[:1]),
        ("lat", info[:, 1]),
        ("lon", info[:, 2]),
        ("elv", info[:, 3]),
        ("num", data[:, 0].astype("int")),
        ("nam", info[:, 0][data[:, 0].astype("int")-1])
        ])

    if "rhophas" in data_type.lower():

        """
         Site      Frequency
         AppRxxCal   PhsxxCal   AppRxyCal   PhsxyCal   AppRyxCal  PhsyxCal  AppRyyCal   PhsyyCal
         AppRxxObs   PhsxxObs   AppRxyObs   PhsxyObs   AppRyxObs  PhsyxObs  AppRyyObs   PhsyyObs
         AppRxxErr   PhsxxErr   AppRxyErr   PhsxyErr   AppRyxErr  PhsyxErr  AppRyyErr   PhsyyErr


        """
        # print(np.shape(data))
        # print(np.shape(data[:, 2:10 ]))
        # print(np.shape(data[:, 10:18 ]))
        # print(np.shape(data[:, 18:26 ]))
        type_dict = dict([
            ("cal", data[:, 2:10]),
            ("obs", data[:, 10:18 ]),
            ("err", data[:, 18:26]),
        ])

    elif "imp" in data_type.lower():

        """
         Site      Frequency
         ReZxxCal   ImZxxCal   ReZxyCal   ImZxyCal   ReZyxCal  ImZyxCal  ReZyyCal   ImZyyCal
         ReZxxObs   ImZxxObs   ReZxyObs   ImZxyObs   ReZyxObs  ImZyxObs  ReZyyObs   ImZyyObs
         ReZxxErr   ImZxxErr   ReZxyErr   ImZxyErr   ReZyxErr  ImZyxErr  ReZyyErr   ImZyyErr
        """

        type_dict = dict([
            ("cal", data[:, 2:10 ]),
            ("obs", data[:, 10:18 ]),
            ("err", data[:, 18:26]),
        ])

    elif "vtf" in data_type.lower():

        """
        Site    Frequency
        ReTzxCal   ImTzxCal   ReTzyCal   ImTzyCal
        ReTzxOb    ImTzxObs   ReTzyObs   ImTzyObs
        ReTzxErr   ImTzxErr   ReTzyErr   ImTzyErr
        """
        type_dict = dict([
            ("cal", data[:, 2:6 ]),
            ("obs", data[:, 6:10 ]),
            ("err", data[:, 10:15]),

        ])

    elif "pt" in data_type.lower():
        """
        Site    Frequency
        ReTzxCal   ImTzxCal   ReTzyCal   ImTzyCal
        ReTzxOb    ImTzxObs   ReTzyObs   ImTzyObs
        ReTzxErr   ImTzxErr   ReTzyErr   ImTzyErr
        """
        type_dict = dict([
            ("cal", data[:, 2:6 ]),
            ("obs", data[:, 6:18 ]),
            ("err", data[:, 10:14]),

        ])

    else:
        error("get_femtic_data: data type "+data_type.lower()+" not implemented! Exit.")

    data_dict = {**head_dict, **type_dict}

    return data_dict
