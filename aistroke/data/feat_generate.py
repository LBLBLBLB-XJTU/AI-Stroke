import torch
import torch.nn.functional as F
import numpy as np

mask = [0,7,8,9,10,11,12,13,14,15,16]

def normalize_smpl_output_torch(joints, root_idx=0, head_idx=10, eps=1e-6):
    B, J, C = joints.shape
    joints = joints.clone()

    # 平移：将 root 节点对齐到原点
    root = joints[:, root_idx:root_idx+1, :]
    joints = joints - root

    # 缩放：用 root→head 的距离当单位长度
    vec = joints[:, head_idx, :] - joints[:, root_idx, :]
    height = torch.norm(vec, dim=-1, keepdim=True).clamp(min=eps)
    joints = joints / height.view(B,1,1)

    # 旋转：旋转：对齐到上方向 Y 轴
    target_up = torch.tensor([0,1,0], dtype=joints.dtype, device=joints.device)
    v = joints[:, head_idx, :] - joints[:, root_idx, :]
    v_norm = F.normalize(v + eps, dim=-1)
    target_up_expand = target_up.unsqueeze(0).expand(B,-1)

    axis = torch.cross(v_norm, target_up_expand, dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    valid = axis_norm[:,0] > eps
    axis_unit = torch.zeros_like(axis)
    axis_unit[valid] = axis[valid] / axis_norm[valid]

    cos_angle = (v_norm * target_up_expand).sum(dim=-1).clamp(-1,1)
    angle = torch.acos(cos_angle)
    K = torch.zeros(B,3,3, device=joints.device)
    K[:,0,1] = -axis_unit[:,2]
    K[:,0,2] = axis_unit[:,1]
    K[:,1,0] = axis_unit[:,2]
    K[:,1,2] = -axis_unit[:,0]
    K[:,2,0] = -axis_unit[:,1]
    K[:,2,1] = axis_unit[:,0]
    I = torch.eye(3, device=joints.device).unsqueeze(0).repeat(B,1,1)
    angle = angle[:,None,None]
    R = I + torch.sin(angle)*K + (1-torch.cos(angle))*torch.bmm(K,K)
    joints = torch.bmm(joints, R.transpose(1,2))

    # -------- 旋转：统一面向前方 (+Z) --------
    # 使用肩膀向量生成水平 forward
    right = joints[:,11] - joints[:,14]  # 右肩 - 左肩
    up = torch.tensor([0,1,0], device=joints.device, dtype=joints.dtype).view(1,3).expand_as(right)
    forward = torch.cross(up, right, dim=-1)  # 水平向前
    forward = F.normalize(forward + eps, dim=-1)
    # 只取 XZ 投影
    forward_xz = forward[:, [0,2]]
    forward_xz_norm = torch.norm(forward_xz, dim=-1, keepdim=True).clamp(min=eps)
    forward_xz = forward_xz / forward_xz_norm
    # 旋转角 theta 使 forward_xz 指向 +Z ([0,1])
    theta = torch.atan2(forward_xz[:,0], forward_xz[:,1])  # [B]
    # Rodrigues 绕 Y 轴旋转矩阵
    cos = torch.cos(-theta)
    sin = torch.sin(-theta)
    R_y = torch.zeros(B,3,3, device=joints.device, dtype=joints.dtype)
    R_y[:,0,0] = cos;  R_y[:,0,1] = 0; R_y[:,0,2] = sin
    R_y[:,1,0] = 0;    R_y[:,1,1] = 1; R_y[:,1,2] = 0
    R_y[:,2,0] = -sin; R_y[:,2,1] = 0; R_y[:,2,2] = cos
    # 应用旋转
    joints = torch.bmm(joints, R_y.transpose(1,2))

    return joints

def compute_limbs_angle_batch(p1,p2,p3,p4,eps=1e-6):
    line1 = p2 - p1
    line2 = p3 - p4
    norm1 = torch.norm(line1, dim=-1)
    norm2 = torch.norm(line2, dim=-1)
    line1_unit = line1 / (norm1[...,None] + eps)
    line2_unit = line2 / (norm2[...,None] + eps)
    cos_theta = torch.sum(line1_unit * line2_unit, dim=-1).clamp(-1,1)
    angle = torch.acos(cos_theta) * (180.0 / np.pi)
    return angle

def generate_feat(cfg, joints, device, modalities_used):
    device = cfg.DEVICE
    joints = torch.tensor(joints, dtype=torch.float32, device=device)
    results = {}
    T = joints.shape[0]

    # visualize_skeleton_3d(joints, "skeleton_before_norm.png")
    joints_all = normalize_smpl_output_torch(joints.to(device))
    # visualize_skeleton_3d(joints_all, "skeleton_after_norm.png")

    if "joints" in modalities_used:
        joints_masked = joints_all[:, mask, :]
        results["joints"] = joints_masked

    if "left_arm_angle" in modalities_used:
        results["left_arm_angle"] = compute_limbs_angle_batch(joints_all[:,11], joints_all[:,13], joints_all[:,11], joints_all[:,4])
    if "right_arm_angle" in modalities_used:
        results["right_arm_angle"] = compute_limbs_angle_batch(joints_all[:,14], joints_all[:,16], joints_all[:,14], joints_all[:,1])

    if "diff" in modalities_used:
        if "left_arm_angle" not in results.keys():
            results["left_arm_angle"] = compute_limbs_angle_batch(joints_all[:,11], joints_all[:,13], joints_all[:,11], joints_all[:,4])
        if "right_arm_angle" not in results.keys():
            results["right_arm_angle"] = compute_limbs_angle_batch(joints_all[:,14], joints_all[:,16], joints_all[:,14], joints_all[:,1])
        results["diff"] = torch.abs(results["left_arm_angle"] - results["right_arm_angle"])

    # 检查所有第一维 T 是否一致
    t_sizes = [v.shape[0] for v in results.values()]
    assert len(set(t_sizes)) == 1 and t_sizes[0] == T, f"results 中 tensor 第一维 (T) 不一致: {t_sizes}"

    return results

def visualize_skeleton_3d(joints, save_path, root_idx=0, head_idx=10, left_shoulder_idx=11, right_shoulder_idx=14, frame_idx=0):
    import numpy as np
    import matplotlib.pyplot as plt

    # 转为 numpy
    if isinstance(joints, (torch.Tensor,)):
        joints = joints.detach().cpu().numpy()

    if joints.ndim == 3:
        joints = joints[frame_idx]

    V = joints.shape[0]

    # --- Y 轴对齐检查 ---
    root = joints[root_idx]
    head = joints[head_idx]
    v = head - root
    v_norm = v / (np.linalg.norm(v)+1e-6)
    print(f"Frame {frame_idx} orientation vector (root->head): {v_norm}")

    # --- 水平朝向检查（左右肩叉上向量） ---
    left_shoulder = joints[left_shoulder_idx]
    right_shoulder = joints[right_shoulder_idx]
    up = np.array([0,1,0])
    forward = np.cross(up, right_shoulder - left_shoulder)
    forward /= (np.linalg.norm(forward)+1e-6)
    print(f"Frame {frame_idx} horizontal forward (shoulders cross up): {forward}")

    # --- 绘图 ---
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    x = joints[:,0]
    y = joints[:,1]
    z = joints[:,2]

    ax.scatter(x, z, y, s=30)  # X水平，Z深度，Y竖直
    for i in range(V):
        ax.text(x[i], z[i], y[i], str(i), color='red')

    # 设置固定坐标轴，不旋转视角
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(mid_y - max_range, mid_y + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')  # 深度
    ax.set_zlabel('Y')  # 高度

    # 固定视角，不旋转
    ax.view_init(elev=0, azim=90)  # 可根据需求调微小角度，但不随数据旋转

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] saved frame {frame_idx} to {save_path}")

