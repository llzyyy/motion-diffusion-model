import numpy as np

# 读取 MDM 生成结果
data = np.load(
    r"./outputs/wave_baseline/results.npy",
    allow_pickle=True
).item()

motion = data["motion"]

print("motion shape:", motion.shape)
print("text:", data["text"])
print("length:", data["lengths"])

# 第一条动作
m = motion[0]                 # [22, 3, T]

# HumanML3D 关节编号
RIGHT_SHOULDER = 17
RIGHT_ELBOW = 19
RIGHT_WRIST = 21

shoulder = m[RIGHT_SHOULDER]  # [3, T]
elbow = m[RIGHT_ELBOW]        # [3, T]
wrist = m[RIGHT_WRIST]        # [3, T]

print("shoulder shape:", shoulder.shape)
print("elbow shape:", elbow.shape)
print("wrist shape:", wrist.shape)

# 手腕相对于肩膀的位置
relative_wrist = wrist - shoulder       # [3, T]

# 转成 [T, 3]
relative_wrist = relative_wrist.T

# 平均位置
center = relative_wrist.mean(axis=0)

# 每一帧距离平均位置多远
distance = np.linalg.norm(
    relative_wrist - center,
    axis=1
)

# RMS 振幅
amplitude_rms = np.sqrt(np.mean(distance ** 2))

# 最大偏移
amplitude_max = np.max(distance)

print()
print("Right wrist RMS amplitude:", amplitude_rms)
print("Right wrist max amplitude:", amplitude_max)