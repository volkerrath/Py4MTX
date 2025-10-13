#!/usr/bin/env python3

import pyMT.data_structures as WSDS
import sys
import pyMT.utils as utils
import pyMT.IO as WSIO


def main(in_file, out_file):
    data = WSDS.Data(in_file)
    # if data.inv_type == 5:
    #     print('Ignoring transfer function data...\n')
    #     data.inv_type = 1
    data.write(outfile=out_file, file_format='modem')


def parse_args(args):
    if len(args) == 1:
        print('Usage is: ws2ModEM <input data> <output data>')
        return False
    elif len(args) == 2:
        in_file = sys.argv[1]
        print('No output specified. Using "ModEM_input.data".\n')
        out_file = 'ModEM_input.data'
    elif len(args) == 3:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
    else:
        print('Too many inputs.\n')
        print('Usage is: ws2ModEM <input data> <output data>\n')
        return

    if not utils.check_file(in_file):
        print('{} not found.\n')
        return False
    if utils.check_file(out_file):
        print('{} already exists.\n'.format(out_file))
        user_response = WSIO.verify_input('Overwrite?', expected='yn', default='y')
        if user_response.lower() == 'n':
            out_file = WSIO.verify_input('Please specify another output file.', expected='write')
    return in_file, out_file


if __name__ == '__main__':
    args = parse_args(sys.argv)
    if args:
        main(in_file=args[0], out_file=args[1])
