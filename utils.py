import csv


def read_point_cloud_csv(filename: str):
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for x, y, z in reader:
            yield float(x), float(y), float(z)


def save_point_cloud_csv(filename: str, x, y, z):
    points = zip(x, y, z)
    with open(filename, 'w', newline='\n') as csv_file:
        csv_writer = csv.writer(csv_file)
        for p in points:
            csv_writer.writerow(p)
