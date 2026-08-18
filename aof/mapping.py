"""
mapping.py — 10 类本体 → FSD50K label 映射（冻结于 2026-08-04, HTML 附录 [待核] 项核验）
每条 label 必须 ∈ FSD50K 200-vocabulary（程序化断言）。诚实标注: FSD50K 无 tire_squeal/
construction/emergency_vehicle 直接标签 → 用近似标签并预期样本量少。
unknown 类(9)恒掩码, 不参与训练/评估。
"""
import csv
import os
from collections import Counter

from . import config as C

# 10 类本体 → FSD50K labels（名字须与 vocabulary.csv 完全一致）
ONTOLOGY = {
    "vehicle": [
        "Vehicle", "Car", "Car_passing_by", "Motor_vehicle_(road)", "Truck", "Bus",
        "Motorcycle", "Race_car_and_auto_racing", "Engine", "Engine_starting", "Idling",
        "Train", "Rail_transport", "Subway_and_metro_and_underground", "Aircraft",
        "Fixed-wing_aircraft_and_airplane", "Boat_and_Water_vehicle",
        "Traffic_noise_and_roadway_noise",
    ],
    "bicycle": ["Bicycle"],
    "horn": ["Vehicle_horn_and_car_horn_and_honking", "Bicycle_bell"],
    "siren": ["Siren", "Alarm"],
    "tire_squeal": ["Squeak", "Screech"],          # 近似标签 [诚实标注: FSD50K 无直接标签]
    "impact": ["Thump_and_thud", "Slam", "Explosion", "Gunshot_and_gunfire", "Knock", "Boom"],
    "construction": ["Power_tool", "Drill", "Sawing", "Hammer", "Tools"],   # 近似 [无 Construction 直接标签]
    "mechanical_anomaly": ["Mechanisms", "Ratchet_and_pawl", "Buzz", "Hiss"],
    "human_activity": [
        "Human_voice", "Speech", "Conversation", "Chatter", "Shout", "Yell", "Screaming",
        "Laughter", "Cough", "Walk_and_footsteps", "Run", "Clapping", "Cheering", "Singing",
        "Female_singing", "Male_singing", "Crowd", "Whispering", "Applause", "Giggle",
        "Chuckle_and_chortle", "Sneeze", "Child_speech_and_kid_speaking",
        "Female_speech_and_woman_speaking", "Male_speech_and_man_speaking",
    ],
}
# 本体类名 → 类别索引（0-8 参与, 9=unknown 掩码）
CLASS_NAMES = list(ONTOLOGY.keys()) + ["unknown"]
CLASS_INDEX = {n: i for i, n in enumerate(CLASS_NAMES)}
assert len(CLASS_NAMES) == C.N_CLASSES

LABEL_TO_CLASS = {}
for name, labels in ONTOLOGY.items():
    for lb in labels:
        LABEL_TO_CLASS[lb] = CLASS_INDEX[name]


def load_vocabulary(fsd50k_dir: str = C.FSD50K_DIR) -> set:
    p = os.path.join(fsd50k_dir, "labels", "vocabulary.csv")
    vocab = set()
    with open(p) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                vocab.add(parts[1])
    return vocab


def verify_mapping(fsd50k_dir: str = C.FSD50K_DIR) -> dict:
    """断言所有映射 label ∈ 200-vocabulary；返回 {class: n_labels}。"""
    vocab = load_vocabulary(fsd50k_dir)
    missing = {lb: cls for cls, labels in ONTOLOGY.items() for lb in labels if lb not in vocab}
    assert not missing, f"映射标签不在 vocabulary: {missing}"
    return {cls: len(labels) for cls, labels in ONTOLOGY.items()}


def build_dev_index(fsd50k_dir: str = C.FSD50K_DIR):
    """dev.csv → [(fname, multi_hot_y(10), split)]。multi-hot: 一个 clip 可属多个本体类。"""
    p = os.path.join(fsd50k_dir, "labels", "dev.csv")
    index = []
    with open(p) as f:
        for r in csv.reader(f):
            if r[0] == "fname":
                continue
            fname, labels, mids, split = r[0], r[1], r[2], r[3].strip()
            y = [0.0] * C.N_CLASSES
            for lb in labels.split(","):
                lb = lb.strip()
                if lb in LABEL_TO_CLASS:
                    y[LABEL_TO_CLASS[lb]] = 1.0
            index.append((fname, y, split))
    return index


def class_distribution(index) -> dict:
    c = Counter()
    for _, y, split in index:
        for i, v in enumerate(y):
            if v and i < C.N_CLASSES - 1:
                c[CLASS_NAMES[i]] += 1
    return dict(c)


def class_distribution_by_split(index) -> dict:
    out = {}
    for split in ("train", "val"):
        c = Counter()
        for _, y, s in index:
            if s != split:
                continue
            for i, v in enumerate(y):
                if v and i < C.N_CLASSES - 1:
                    c[CLASS_NAMES[i]] += 1
        out[split] = dict(c)
    return out
