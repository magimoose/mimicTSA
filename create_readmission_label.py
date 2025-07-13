import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm
import re

# modified from mimic3-benchmarks


def process_partition(args, partition, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        unsorted_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        patient_ts_files = sorted(unsorted_ts_files, key=lambda x: int(x.split('episode')[1].split('_')[0]))
        
        patient_stays_file = os.path.join(patient_folder, "stays.csv")
        stays_df = pd.read_csv(patient_stays_file).sort_values(by='INTIME')

        stays_df['INTIME'] = pd.to_datetime(stays_df["INTIME"])
        stays_df['OUTTIME'] = pd.to_datetime(stays_df["OUTTIME"])

        for idx, ts_filename in enumerate(patient_ts_files, start=0):
            # print(idx, ts_filename)
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    # print("\n\t(empty label file)", patient, ts_filename)
                    continue

                

                if idx + 1 in stays_df.index:
                    current_outtime = stays_df.loc[idx, 'OUTTIME']
                    next_intime = stays_df.loc[idx + 1, 'INTIME']
                    days_diff = (next_intime - current_outtime).days
                    if days_diff < 30:
                        readmissionWithin30 = 1
                    else:
                        readmissionWithin30 = 0
                else:
                    readmissionWithin30 = 0
                    
                mortality = int(label_df.iloc[0]["Mortality"])
                if mortality == 1:
                    continue
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    # print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                # if los < n_hours - eps:
                #     print("\n\t(too short)", patient, ts_filename)
                #     continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                # ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                #             if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    # print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        clean_line = re.sub(r'[a-zA-Z \-/]', '', line)
                        outfile.write(clean_line)
                # print(output_ts_filename)
                xy_pairs.append((output_ts_filename, readmissionWithin30))

    print("Number of created samples:", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    # if partition == "test":
    #     xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in xy_pairs:
            listfile.write('{},{:d}\n'.format(x, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
