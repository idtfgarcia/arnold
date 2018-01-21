__author__ = 'Ivan'

import csv as csv
import numpy as np
import sys as sys

class ioUtils:

    @staticmethod
    def array2csv(fullpath,table,delimiter=';'):

        print('Data type: ' + str(type(table)))
        with open(fullpath, 'w') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n', delimiter=delimiter)

            if isinstance(table, np.ndarray):

                if table.ndim == 2:
                    print('Attempting to write file: ' + fullpath)
                    [writer.writerow(r) for r in table]
                elif table.ndim == 1:
                    print('Attempting to write file: ' + fullpath)
                    writer.writerow(table)
                else:
                    sys.exit('Cannot write csv from (>2)D array')

            elif isinstance(table, list):
                print('Attempting to write file: ' + fullpath)
                [writer.writerow(r) for r in table]

        print('File written successfully.')

    @staticmethod
    def csv2array(fullpath,delimiter=';'):

        with open(fullpath,'r') as csvfile:
            reader = csv.reader(csvfile, lineterminator='\n', delimiter=delimiter)
            table = [[float(str(e).strip()) for e in r] for r in reader]

        table = np.array(table, dtype=float)
        #table = table.transpose()
        print('File succesfully read.')
        return table;

