import numpy as np
import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import torch
import joblib

DATA_ROOT_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "data", "TRSPD")
DATA_WITH_ANGLES_PATH = osp.join(DATA_ROOT_DIR, "trspd_data.pkl")

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_data_trspd(mask):
    os.chdir(osp.join(DATA_ROOT_DIR, 'data_new'))
    subjects = [name for name in os.listdir(".") if os.path.isdir(name)]

    joints_sel = mask
    total_joint_num = 25
    invert_data = True

    Hnames, Hdatas = [], []
    Pnames, Pdatas = [], []

    sample_id = 0

    for sub in sorted(subjects):
        print('===========================' + sub + '============================')
        sub_dir = osp.join(DATA_ROOT_DIR, 'data_new', sub)
        tasks = [t for t in os.listdir(sub_dir) if os.path.isdir(osp.join(sub_dir, t))]

        for task in sorted(tasks):
            print(f'  task: {task}')
            fname = osp.join(sub_dir, task, 'Joint_Positions.csv')

            if not os.path.isfile(fname):
                print(f'  {task} does not exist')
                continue

            data_loaded = np.loadtxt(fname, delimiter=",")
            frame_num = len(data_loaded) // total_joint_num

            features = np.zeros((frame_num, len(joints_sel), 3), dtype=np.float32)

            for index, row in enumerate(data_loaded):
                f = index // total_joint_num
                r = index % total_joint_num

                if r in joints_sel:
                    j = joints_sel.index(r)

                    x = -row[0] if (invert_data and 'L' in task) else row[0]
                    y = row[2]
                    z = row[1]

                    features[f, j] = [x, y, z]

            sample_name = f"{sub}_{sample_id}"
            sample_id += 1

            if sub[0] == 'H':
                Hnames.append(sample_name)
                Hdatas.append(features)
            elif sub[0] == 'P':
                Pnames.append(sample_name)
                Pdatas.append(features)

    return {
        "Hnames": Hnames,
        "Hdatas": Hdatas,
        "Pnames": Pnames,
        "Pdatas": Pdatas
    }


def compute_angle(start1, end1, start2, end2):
	start1 = torch.tensor(start1, dtype=torch.float32) if isinstance(start1, np.ndarray) else start1
	end1 = torch.tensor(end1, dtype=torch.float32) if isinstance(end1, np.ndarray) else end1
	start2 = torch.tensor(start2, dtype=torch.float32) if isinstance(start2, np.ndarray) else start2
	end2 = torch.tensor(end2, dtype=torch.float32) if isinstance(end2, np.ndarray) else end2
	
	vec1 = start1 - end1
	vec2 = start2 - end2

	norm1 = torch.norm(vec1)
	norm2 = torch.norm(vec2)

	if norm1 < 1e-6 or norm2 < 1e-6:
		return 0.0

	vec1_unit = vec1 / norm1
	vec2_unit = vec2 / norm2
	cos_theta = torch.clamp(torch.dot(vec1_unit, vec2_unit), -1.0, 1.0)

	angle_rad = torch.acos(cos_theta)
	angle_deg = torch.rad2deg(angle_rad)

	return angle_deg.item()

def get_angles(label_data):
    new_data = []

    for i in range(len(label_data["Hnames"])):
        hname = label_data["Hnames"][i]
        joints = label_data["Hdatas"][i]

        left_angles = []
        right_angles = []

        for joint_frame in joints:
            left_angles.append(
				compute_angle(joint_frame[0], joint_frame[2], joint_frame[4], joint_frame[6])
			)
            right_angles.append(
				compute_angle(joint_frame[0], joint_frame[2], joint_frame[8], joint_frame[10])
			)
        
        if hname[0] == 'H':
            label = 1

        sample = {
            "id": hname,
            "joints": joints,
            "left_angles": left_angles,
            "right_angles": right_angles,
            "total_label": label
        }

        new_data.append(sample)
        
    for i in range(len(label_data["Pnames"])):
        pname = label_data["Pnames"][i]
        joints = label_data["Pdatas"][i]

        left_angles = []
        right_angles = []

        for joint_frame in joints:
            left_angles.append(
				compute_angle(joint_frame[0], joint_frame[2], joint_frame[4], joint_frame[6])
			)
            right_angles.append(
				compute_angle(joint_frame[0], joint_frame[2], joint_frame[8], joint_frame[10])
			)
        
        if pname[0] == 'P':
            label = 0

        sample = {
            "id": pname,
            "joints": joints,
            "left_angles": left_angles,
            "right_angles": right_angles,
            "total_label": label
        }

        new_data.append(sample)
        
    joblib.dump(new_data, DATA_WITH_ANGLES_PATH)
    
if __name__ == "__main__":
    UPPER_MASK = [0,1,2,3,4,5,6,7,8,9,10,11,20,21,22,23,24]
    label_data = load_data_trspd(UPPER_MASK)
    get_angles(label_data)

def vis_one_frame_trspd(data_file):
    # ===== 1. Kinect bone definition =====
    KINECT_BONES = [
        (0, 1), (1, 20), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7),
        (7, 21), (6, 22),
        (20, 8), (8, 9), (9, 10), (10, 11),
        (11, 23), (10, 24),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19),
    ]

    # ===== 2. Load your data =====
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    # 取：第一个健康被试的第一帧
    frame = data["Hdatas"][0][0]  # shape (75,)

    num_joints = 25

    # 注意你代码里的顺序：x, z, y
    x = frame[0:num_joints]
    z = frame[num_joints:2*num_joints]
    y = frame[2*num_joints:3*num_joints]

    # ===== 3. Plot =====
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # joints
    ax.scatter(x, y, z, c='red', s=40)

    # bones
    for i, j in KINECT_BONES:
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            [z[i], z[j]],
            c='black',
            linewidth=2
        )

    # joint index labels
    for i in range(num_joints):
        ax.text(x[i], y[i], z[i], str(i), fontsize=9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Kinect Skeleton (JointType Order 0–24)")
    ax.view_init(elev=0, azim=90)

    plt.tight_layout()
    plt.savefig(osp.join(osp.dirname(data_file), "vis_skeleton.png"))