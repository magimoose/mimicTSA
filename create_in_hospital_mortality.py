import os
import argparse
import pandas as pd
import random
random.seed(49297)
from tqdm import tqdm
import re
from nltk import sent_tokenize, word_tokenize

#modified from mimic3-benchmarks with elements from ClinicalNotesICU

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)

def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))

def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)

def clean_text(text):
    """
    Clean text
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text

def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section

def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            text = ' '.join(word_tokenize(sent))
            yield text.lower()

def getText(t):
    return " ".join(list(preprocess_mimic(t)))

def process_partition(args, partition, df, eps=1e-6, n_hours=48):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        notes_df = df[df.SUBJECT_ID == int(patient)]
        if notes_df.shape[0] == 0:
            print("No notes for PATIENT_ID : {}".format(int(patient)))
            continue
        notes_df.sort_values(by='CHARTTIME', inplace=True)
        notes_df['CHARTTIME'] = pd.to_datetime(notes_df['CHARTTIME'])

        stays_path = os.path.join(patient_folder, 'stays.csv')
        stays_df = pd.read_csv(stays_path)
        stays_df.INTIME = pd.to_datetime(stays_df.INTIME)
        stays_df.OUTTIME = pd.to_datetime(stays_df.OUTTIME)
        hadm_ids = list(stays_df.HADM_ID.values)

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                icu_id = label_df.iloc[0]["ICUSTAY_ID"]
                icu_stay_df = stays_df[stays_df.ICUSTAY_ID == icu_id]
                intime = icu_stay_df.INTIME.iloc[0]
                outtime = icu_stay_df.OUTTIME.iloc[0]

                notes = notes_df[(notes_df.HADM_ID.isin(hadm_ids)) & (notes_df.CHARTTIME >= intime) & (notes_df.CHARTTIME <= outtime)]
                if notes.shape[0] == 0:
                    print("\n\t(no notes in ICU) ", patient, ts_filename)
                    continue
                notes = notes["TEXT"].apply(getText).tolist()

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])
                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("\n\t(length of stay is missing)", patient, ts_filename)
                    continue

                if los < n_hours - eps:
                    continue

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < n_hours + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                xty.append((output_ts_filename, notes, mortality))

    print("Number of created samples:", len(xty))
    if partition == "train":
        random.shuffle(xty)
    if partition == "test":
        xty = sorted(xty)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, t, y) in xty:
            listfile.write('{},{:d}\n'.format(x, t, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    df = pd.read_csv('/home/magnusjg/1TB/database/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv')
    df.CHARTDATE = pd.to_datetime(df.CHARTDATE)
    df.CHARTTIME = pd.to_datetime(df.CHARTTIME)
    df.STORETIME = pd.to_datetime(df.STORETIME)
    df = df[df.SUBJECT_ID.notnull()]
    df = df[df.HADM_ID.notnull()]
    df = df[df.CHARTTIME.notnull()]
    df = df[df.TEXT.notnull()]
    df = df[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]

    process_partition(args, "test", df)
    process_partition(args, "train", df)


if __name__ == '__main__':
    main()
