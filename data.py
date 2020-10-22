from pathlib import Path

import numpy as np
import pandas as pd
from numpy.ma import masked_invalid
from scipy.io import loadmat
import argparse


def match_subject(subject: str):
    return int(subject.split(".")[0])


def subject_dir_search(dir: Path, sub_id: str):
    for file in dir.iterdir():
        cur_id = match_subject(file.name)
        if cur_id == sub_id:
            yield file


def file_for_subject(dir: Path, subject_postfix: str):
    matches = list(subject_dir_search(dir, subject_postfix))
    assert (
        len(matches) >
        0), f"subject_postfix: {subject_postfix} has no matches in {str(dir)}"
    assert (
        len(matches) == 1
    ), f"subject_postfix: {subject_postfix} matches: {matches} dir: {str(dir)}"
    return matches[0]


LABEL_MAP = {"Non-Suicidal": 0, "Suicidal": 1, "Control": 2}


def read_static_data(path: Path,
                     labels_path: Path,
                     include: set,
                     scores=False,
                     skip_control=True):
    label_dict = {}
    score_dict = {}
    df = pd.read_csv(labels_path, sep=";")
    for _, row in df.iterrows():
        subject_postfix = row["Subj_ID"]
        label_txt = row["Suicidality_status"]
        if label_txt == "Control" and skip_control:
            continue
        label_dict[subject_postfix] = LABEL_MAP[label_txt]
        score_dict[subject_postfix] = row["SBQ-R"]

    patients_data = []
    patients_labels = []
    patients_scores = []
    for matfile in (path / "FC").glob("*.mat"):
        single_patient_data = []
        if "FC" in include:
            mat_dict = loadmat(matfile)
            fc_data = mat_dict["ROICorrelation_FisherZ"]
            fc_upper_data = fc_data[np.triu_indices_from(fc_data, k=1)]
            single_patient_data.append(fc_upper_data.reshape(-1))
        subject_postfix = match_subject(matfile.name)
        for data_type in ["REHO", "ALFF", "fALFF"]:
            if data_type in include:
                mat_dict = loadmat(
                    str(file_for_subject(path / data_type, subject_postfix)))
                single_patient_data.append(mat_dict["ROISignals"].reshape(-1))
        single_patient_data = np.concatenate(single_patient_data)
        if subject_postfix in label_dict:
            patients_data.append(single_patient_data)
            patients_labels.append(label_dict[subject_postfix])
            patients_scores.append(score_dict[subject_postfix])
        else:
            print(f"Skipping {subject_postfix}")
    X = np.stack(patients_data, axis=0)
    y = np.stack(patients_labels, axis=0)
    s = np.stack(patients_scores, axis=0)
    # replace infinities with zeros
    X[masked_invalid(X).mask] = 0
    # if all included
    # first 264 -> FC
    # 265th -> REHO
    # 266th -> ALFF
    # 267th -> fALFF
    if scores:
        return X.reshape(X.shape[0], -1), y, s
    else:
        return X.reshape(X.shape[0], -1), y


def read_dynamic_data(path: Path,
                      labels_path: Path,
                      include: set,
                      scores=False,
                      skip_control=True,
                      return_ids=False):
    include = {f"d{t}" for t in include}

    label_dict = {}
    score_dict = {}
    df = pd.read_csv(labels_path, sep=";")
    for _, row in df.iterrows():
        subject_postfix = row["Subj_ID"]
        label_txt = row["Suicidality_status"]
        if label_txt == "Control" and skip_control:
            print(f'Skipping patient {subject_postfix} from control group')
            continue
        label_dict[subject_postfix] = LABEL_MAP[label_txt]
        score_dict[subject_postfix] = row["SBQ-R"]

    patients_data = []
    patients_labels = []
    patients_scores = []
    patients_ids = []
    for matfile in (path / "dFC").glob("*.mat"):
        single_patient_data = []
        if "dFC" in include:
            print(f"Loading {str(matfile)}")
            mat_dict = loadmat(matfile)
            key = "mat"
            atlas_size = mat_dict[key].shape[1]
            assert mat_dict[key].shape == (341, atlas_size, atlas_size)
            fcs = []
            for i in range(mat_dict[key].shape[0]):
                fc_data = mat_dict[key][i]
                fc_upper_data = fc_data[np.triu_indices_from(fc_data, k=1)]
                fcs.append(fc_upper_data)
            single_patient_data.append(np.stack(fcs))
        subject_postfix = match_subject(matfile.name)
        for data_type in ["dREHO", "dALFF", "dfALFF"]:
            if data_type in include:
                mat_path = file_for_subject(path / data_type, subject_postfix)
                print(f"Loading {str(mat_path)}")
                mat_dict = loadmat(str(mat_path))
                key = "ROISignals"
                single_patient_data.append(mat_dict[key])
        single_patient_data = np.concatenate(single_patient_data, axis=1)
        # print(f'single_patient_data.shape: {single_patient_data.shape}')
        if subject_postfix in label_dict:
            patients_data.append(single_patient_data)
            patients_labels.append(label_dict[subject_postfix])
            patients_scores.append(score_dict[subject_postfix])
            patients_ids.append(subject_postfix)
        else:
            print(f"Skipping {subject_postfix}")
    X = np.stack(patients_data, axis=0)
    y = np.stack(patients_labels, axis=0)
    s = np.stack(patients_scores, axis=0)
    ids = np.stack(patients_ids, axis=0)
    # replace infinities with zeros
    X[masked_invalid(X).mask] = 0
    # if all included
    # first 264 -> FC
    # 265th -> REHO
    # 266th -> ALFF
    # 267th -> fALFF
    if return_ids:
        if scores:
            return X, y, s, ids
        else:
            return X, y, ids
    else:
        if scores:
            return X, y, s
        else:
            return X, y


def read_raw_data(path: Path):
    key = "ROISignals"
    patients_data = []
    for matfile in (path).glob("*.mat"):
        mat_dict = loadmat(matfile)
        data = mat_dict[key]
        # print(f'data shape: {mat_dict[key].shape}')
        assert mat_dict[key].shape == (390, 264)
        patients_data.append(data)
    data = np.stack(patients_data, axis=0)
    return data.reshape(data.shape[0], -1)


def main():
    included = set(args.include)
    included = ("all" if included == {"FC", "REHO", "ALFF", "fALFF"} else
                "_".join(sorted(included)))

    assert 'dynamic' in str(args.data).lower()
    X, l, y, ids = read_dynamic_data(args.data,
                                     args.labels,
                                     args.include,
                                     scores=True,
                                     skip_control=not args.with_control,
                                     return_ids=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="directory with data")
    parser.add_argument("labels", type=Path, help="csv file with labels")
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        help="data to include: 'FC', 'REHO', 'ALFF', 'fALFF'",
        default=["FC", "REHO", "ALFF", "fALFF"],
    )
    parser.add_argument("--with_control",
                        action="store_true",
                        help="include the control group")
    args = parser.parse_args()
    main()
