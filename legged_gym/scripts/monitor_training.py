"""
dog_v2 HIMLoco 训练监控脚本
读取 TensorBoard 日志，分析奖励趋势，检测异常。
用法: .venv/bin/python legged_gym/scripts/monitor_training.py
"""
import os
import sys
import glob
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# 配置
# ============================================================
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs", "flat_dog_v2")

# 需要监控的关键指标
KEY_METRICS = [
    ("Train/mean_reward", "mean_reward"),
    ("Train/mean_episode_length", "mean_episode_length"),
    ("Loss/value_function", "value_loss"),
    ("Loss/surrogate", "surrogate_loss"),
    ("Loss/Estimation Loss", "estimation_loss"),
    ("Loss/Swap Loss", "swap_loss"),
    ("Loss/learning_rate", "learning_rate"),
    ("Policy/mean_noise_std", "noise_std"),
]

# 需要从 Episode/ 前缀中提取的奖励分量
EPISODE_REWARDS = [
    "rew_tracking_lin_vel",
    "rew_tracking_ang_vel",
    "rew_base_height",
    "rew_lin_vel_z",
    "rew_orientation",
    "rew_dof_acc",
    "rew_joint_power",
    "rew_action_rate",
    "rew_smoothness",
    "rew_feet_air_time",
    "rew_stand_still",
    "rew_collision",
    "rew_default_pos_linear",
    "rew_diagonal_sync",
    "rew_hip_mirror_symmetry",
]

# 异常检测阈值
WARN_EPISODE_LEN_SHORT = 200     # episode 长度持续低于此值
WARN_REWARD_DECLINE_RATE = 0.5   # 最近100轮奖励下降超过此比例
WARN_LOSS_EXPLODE = 100.0        # loss 超过此值
WARN_NAN = True                  # 检测 NaN

# 趋势分析窗口
TREND_WINDOW = 100               # 最近100轮用于趋势分析
TREND_COMPARE_WINDOW = 100       # 与前100轮对比


def find_latest_log_dir():
    """找到最新的训练日志目录"""
    if not os.path.isdir(LOG_ROOT):
        return None
    dirs = sorted(glob.glob(os.path.join(LOG_ROOT, "*dog_v2_himloco*")))
    if not dirs:
        return None
    return dirs[-1]


def load_events(log_dir):
    """加载 TensorBoard 事件数据"""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    return ea


def get_scalar(ea, tag):
    """获取标量数据"""
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except (KeyError, IndexError):
        return [], []


def check_nan(values):
    """检测 NaN"""
    for i, v in enumerate(values):
        if np.isnan(v) or np.isinf(v):
            return True, i, v
    return False, -1, None


def compute_trend(values):
    """计算趋势: 用最近 TREND_WINDOW 个点的线性回归斜率"""
    if len(values) < TREND_WINDOW:
        return 0.0, 0.0
    recent = np.array(values[-TREND_WINDOW:])
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    mean = np.mean(recent)
    return slope, mean


def analyze(ea):
    """分析训练状态，返回 (status, issues)"""
    issues = []
    status = {}

    # 1. 检查关键指标
    for tag, key in KEY_METRICS:
        steps, values = get_scalar(ea, tag)
        if not values:
            continue

        has_nan, nan_idx, nan_val = check_nan(values)
        if has_nan and WARN_NAN:
            issues.append(f"[CRITICAL] {key}: NaN detected at step {nan_idx if nan_idx >= 0 else '?'}")

        slope, mean = compute_trend(values)
        status[key] = {"mean": mean, "slope": slope, "latest": values[-1], "total_points": len(values)}

        # Loss 爆炸检测
        if "loss" in key.lower() and key != "estimation_loss" and key != "swap_loss":
            if mean > WARN_LOSS_EXPLODE:
                issues.append(f"[WARN] {key}: mean={mean:.2f} > {WARN_LOSS_EXPLODE} (可能爆炸)")

    # 2. 检查 episode 长度
    ep_len = status.get("mean_episode_length")
    if ep_len:
        _, ep_len_values = get_scalar(ea, "Train/mean_episode_length")
        recent_len = np.mean(ep_len_values[-TREND_WINDOW:]) if len(ep_len_values) >= TREND_WINDOW else np.mean(ep_len_values)
        if recent_len < WARN_EPISODE_LEN_SHORT:
            issues.append(f"[WARN] episode_length 均值={recent_len:.0f} < {WARN_EPISODE_LEN_SHORT} (机器人频繁终止)")

    # 3. 检查奖励趋势
    reward = status.get("mean_reward")
    if reward and len(get_scalar(ea, "Train/mean_reward")[1]) > TREND_WINDOW + TREND_COMPARE_WINDOW:
        _, reward_values = get_scalar(ea, "Train/mean_reward")
        recent = np.mean(reward_values[-TREND_WINDOW:])
        previous = np.mean(reward_values[-(TREND_WINDOW + TREND_COMPARE_WINDOW):-TREND_WINDOW])
        if previous != 0:
            decline = (previous - recent) / abs(previous)
            if decline > WARN_REWARD_DECLINE_RATE and recent < 0:
                issues.append(f"[WARN] mean_reward 下降 {decline*100:.0f}% (前={previous:.2f}, 现={recent:.2f})")

    # 4. 检查各奖励分量
    for rew_key in EPISODE_REWARDS:
        tag = f"Episode/{rew_key}"
        steps, values = get_scalar(ea, tag)
        if not values:
            continue
        has_nan, _, _ = check_nan(values)
        if has_nan:
            issues.append(f"[CRITICAL] {rew_key}: NaN detected!")
        slope, mean = compute_trend(values)
        status[f"ep_{rew_key}"] = {"mean": mean, "slope": slope, "latest": values[-1]}

    # 5. 检查 noise_std 收敛情况
    noise = status.get("noise_std")
    if noise and noise["total_points"] > 1000:
        if noise["mean"] < 0.1:
            issues.append(f"[INFO] noise_std={noise['mean']:.4f} 已收敛到很小的值 (可能过度确定性)")

    return status, issues


def print_report(log_dir, status, issues):
    """打印报告"""
    print("=" * 60)
    print(f"训练监控报告: {os.path.basename(log_dir)}")
    print("=" * 60)

    # 总览
    reward = status.get("mean_reward")
    ep_len = status.get("mean_episode_length")
    iterations = reward["total_points"] if reward else 0

    print(f"\n[总览]")
    print(f"  迭代轮数: {iterations}")
    if reward:
        print(f"  平均奖励: {reward['latest']:.4f} (趋势: {'+' if reward['slope'] > 0 else ''}{reward['slope']:.6f}/iter)")
    if ep_len:
        print(f"  Episode长度: {ep_len['latest']:.1f} / {4000:.0f}")

    # 损失
    print(f"\n[Loss]")
    for key in ["value_loss", "surrogate_loss", "estimation_loss", "swap_loss"]:
        if key in status:
            s = status[key]
            print(f"  {key:25s}: {s['latest']:.4f} (趋势: {'+' if s['slope'] > 0 else ''}{s['slope']:.6f})")

    # 关键奖励分量
    print(f"\n[奖励分量 (Episode)]")
    # 只打印非零趋势的
    for rew_key in EPISODE_REWARDS:
        ep_key = f"ep_{rew_key}"
        if ep_key in status:
            s = status[ep_key]
            trend_mark = ""
            if abs(s["slope"]) > 1e-5:
                trend_mark = f" {'↑' if s['slope'] > 0 else '↓'}"
            print(f"  {rew_key:30s}: {s['latest']:+.4f}{trend_mark}")

    # 问题
    if issues:
        print(f"\n[问题] 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n[问题] 无异常")

    print("=" * 60)

    return len([i for i in issues if "CRITICAL" in i]) == 0


def main():
    log_dir = find_latest_log_dir()
    if not log_dir:
        print(f"[ERROR] 未找到日志目录: {LOG_ROOT}")
        sys.exit(1)

    print(f"读取日志: {log_dir}")
    ea = load_events(log_dir)

    status, issues = analyze(ea)
    healthy = print_report(log_dir, status, issues)

    # 同时检查 tmux 里训练是否还活着
    import subprocess
    result = subprocess.run(["tmux", "has-session", "-t", "dog_v2_train"], capture_output=True)
    tmux_alive = result.returncode == 0
    print(f"\ntmux session 'dog_v2_train': {'运行中' if tmux_alive else '已停止'}")

    if not tmux_alive:
        print("[WARN] 训练进程已停止!")
    elif not healthy:
        print("[ACTION] 检测到严重问题，建议终止训练并调整参数")

    return 0 if (healthy and tmux_alive) else 1


if __name__ == "__main__":
    sys.exit(main())
