"""
Epsilon Greedy 多臂老虎机学习演示 - Flask 后端
实现 Epsilon Greedy 算法逻辑，提供 REST API 供前端调用
"""

from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# ==================== 全局游戏状态 ====================
game_state = {
    'num_arms': 3,
    'true_probs': [0.8, 0.5, 0.3],
    'epsilon': 0.1,
    'q_values': [0.0, 0.0, 0.0],
    'arm_counts': [0, 0, 0],
    'total_steps': 0,
    'explore_count': 0,
    'exploit_count': 0,
    'total_reward': 0,
    'history': []
}


def reset_state():
    """重置算法状态（保留臂数量、真实概率和 epsilon）"""
    n = game_state['num_arms']
    game_state['q_values'] = [0.0] * n
    game_state['arm_counts'] = [0] * n
    game_state['total_steps'] = 0
    game_state['explore_count'] = 0
    game_state['exploit_count'] = 0
    game_state['total_reward'] = 0
    game_state['history'] = []


def get_best_arm():
    """获取当前估计概率最大的臂编号（多个最大值时随机选一个）"""
    max_q = max(game_state['q_values'])
    # 找到所有最大值的臂
    best_arms = [i for i, q in enumerate(game_state['q_values']) if q == max_q]
    return random.choice(best_arms)


def check_converged():
    """检查算法是否收敛：所有臂的估计概率与真实概率误差均小于 0.05"""
    for i in range(game_state['num_arms']):
        if game_state['arm_counts'][i] == 0:
            return False
        if abs(game_state['q_values'][i] - game_state['true_probs'][i]) >= 0.05:
            return False
    return True


# ==================== 路由定义 ====================

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')


@app.route('/api/init', methods=['POST'])
def api_init():
    """初始化环境：设置臂数量、真实概率和 epsilon"""
    data = request.get_json()
    num_arms = data.get('num_arms', 3)
    true_probs = data.get('true_probs', [0.8, 0.5, 0.3])
    epsilon = data.get('epsilon', 0.1)

    # 参数校验
    num_arms = max(2, min(10, int(num_arms)))
    epsilon = max(0.0, min(1.0, float(epsilon)))
    true_probs = [max(0.0, min(1.0, float(p))) for p in true_probs[:num_arms]]

    # 如果概率数量不足，补齐为 0.5
    while len(true_probs) < num_arms:
        true_probs.append(0.5)

    game_state['num_arms'] = num_arms
    game_state['true_probs'] = true_probs
    game_state['epsilon'] = epsilon
    reset_state()

    return jsonify({
        'success': True,
        'state': get_status_dict()
    })


@app.route('/api/step', methods=['POST'])
def api_step():
    """执行一步 Epsilon Greedy 算法"""
    num_arms = game_state['num_arms']
    epsilon = game_state['epsilon']

    # 记录选择前的 Q 值（用于前端显示更新过程）
    q_before = game_state['q_values'][:]

    # 选择动作：epsilon 概率探索，1-epsilon 概率利用
    if random.random() < epsilon:
        # 探索：随机选择一个臂
        chosen_arm = random.randint(0, num_arms - 1)
        is_explore = True
        game_state['explore_count'] += 1
    else:
        # 利用：选择当前 Q 值最大的臂
        chosen_arm = get_best_arm()
        is_explore = False
        game_state['exploit_count'] += 1

    # 获得奖励：根据真实概率随机产生 1 或 0
    reward = 1 if random.random() < game_state['true_probs'][chosen_arm] else 0

    # 更新估计 Q 值：Q(a) = (Q(a) * N + reward) / (N + 1)
    old_count = game_state['arm_counts'][chosen_arm]
    game_state['q_values'][chosen_arm] = (
        (game_state['q_values'][chosen_arm] * old_count + reward) / (old_count + 1)
    )
    game_state['arm_counts'][chosen_arm] = old_count + 1

    # 更新全局统计
    game_state['total_steps'] += 1
    game_state['total_reward'] += reward

    # 记录历史
    game_state['history'].append({
        'step': game_state['total_steps'],
        'arm': chosen_arm,
        'is_explore': is_explore,
        'reward': reward,
        'q_before': q_before[chosen_arm],
        'q_after': game_state['q_values'][chosen_arm]
    })

    # 检查收敛
    converged = check_converged()

    # 获取当前最佳臂
    best_arm = get_best_arm()

    return jsonify({
        'step': game_state['total_steps'],
        'arm': chosen_arm,
        'is_explore': is_explore,
        'reward': reward,
        'q_values': game_state['q_values'][:],
        'arm_counts': game_state['arm_counts'][:],
        'total_steps': game_state['total_steps'],
        'explore_count': game_state['explore_count'],
        'exploit_count': game_state['exploit_count'],
        'total_reward': game_state['total_reward'],
        'best_arm': best_arm,
        'best_q': game_state['q_values'][best_arm],
        'q_before': q_before[chosen_arm],
        'q_after': game_state['q_values'][chosen_arm],
        'converged': converged
    })


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """重置环境：清除历史记录，重置估计概率和计数"""
    reset_state()
    return jsonify({
        'success': True,
        'state': get_status_dict()
    })


@app.route('/api/status', methods=['GET'])
def api_status():
    """获取当前完整状态"""
    return jsonify(get_status_dict())


def get_status_dict():
    """构造完整状态字典"""
    best_arm = get_best_arm() if game_state['total_steps'] > 0 else 0
    return {
        'num_arms': game_state['num_arms'],
        'true_probs': game_state['true_probs'][:],
        'epsilon': game_state['epsilon'],
        'q_values': game_state['q_values'][:],
        'arm_counts': game_state['arm_counts'][:],
        'total_steps': game_state['total_steps'],
        'explore_count': game_state['explore_count'],
        'exploit_count': game_state['exploit_count'],
        'total_reward': game_state['total_reward'],
        'best_arm': best_arm,
        'best_q': game_state['q_values'][best_arm] if game_state['total_steps'] > 0 else 0,
        'history': game_state['history'][:],
        'converged': check_converged() if game_state['total_steps'] > 0 else False
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
