import fnmatch
import os
import csv
import numpy as np



def write_stats(path, errors):
    folders = list(os.walk(path))[0][1]
    stats = np.array([(_get_lines(folder, path), folder) for folder
                      in folders if os.path.exists(os.path.join(path, folder, CSV_FILE_NAME))])

    paths = get_xml_path(path)
    stats = [_get_lines(path) for path in paths]

    if list(stats):
        # with open(os.path.join(path, 'logs.txt'), 'w') as f:
        #     avg_line_iu = np.average([np.float32(line[0][1][4]) for line in stats])
        #     print('Average lineUI: {}'.format(avg_line_iu))
        #     f.write("\n--------------------------------------------------\n\n")
        #     f.write('Average lineUI: {}'.format(avg_line_iu))

        with open(os.path.join(path, 'summary.csv'), 'w') as f:
            for i, line in enumerate(stats):
                if i == 0:
                    f.write('filename,' + ','.join(line[0]) + '\n')
                f.write(line[1][0] + ',' + ','.join(line[1][1:]) + '\n')

    if not errors:
        return

    with open(os.path.join(path, 'error_log.txt'), 'w') as f:
        for error in errors:
            f.writelines([line.decode('ascii') for line in error[1]])
            f.write("\n--------------------------------------------------\n\n")


def _get_lines(path):
    csv_path = path
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)


def get_xml_path(folder):
    files = []
    for the_file in os.listdir(folder):
        if fnmatch.fnmatch(the_file, '*.csv'):
            files.append(os.path.join(folder, the_file))
    return files

if __name__ == '__main__':
    write_stats(path='./../../res/mathias/', errors=[])
