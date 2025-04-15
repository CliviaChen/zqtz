import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 超参数调整 =====
POP_SIZE = 100
MAX_ITER = 500
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.4
RISK_FREE_RATE = 0.02
RESULT_DIR = "优化投资组合结果"
os.makedirs(RESULT_DIR, exist_ok=True)


# ===== 改进的权重修复函数 =====
def repair_weights(weights, min_w, max_w):
    """基于剩余空间分配的修复策略"""
    weights = np.clip(weights, min_w, max_w)
    sum_weights = np.sum(weights)

    if abs(sum_weights - 1.0) > 1e-6:
        delta = 1.0 - sum_weights
        candidates = np.where((weights < max_w) & (weights > min_w))[0]

        if len(candidates) > 0:
            if delta > 0:
                available = max_w - weights[candidates]
                total_available = np.sum(available)
                if total_available > 1e-6:
                    add = min(delta, total_available)
                    weights[candidates] += add * (available / total_available)
            else:
                available = weights[candidates] - min_w
                total_available = np.sum(available)
                if total_available > 1e-6:
                    subtract = min(-delta, total_available)
                    weights[candidates] -= subtract * (available / total_available)

    weights = np.clip(weights, min_w, max_w)
    return weights / np.sum(weights)


# ===== 增强的初始化方法 =====
def initialize_population(n_assets):
    """拉丁超立方抽样初始化"""
    population = np.zeros((POP_SIZE, n_assets))
    for i in range(n_assets):
        permutation = np.random.permutation(POP_SIZE)
        population[:, i] = permutation / POP_SIZE

    population = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * population
    row_sums = population.sum(axis=1)
    population = (population.T / row_sums).T
    return np.clip(population, MIN_WEIGHT, MAX_WEIGHT)


# ===== 改进的目标函数 =====
def calculate_sharpe(weights, mean_returns, cov_matrix):
    """带惩罚项的夏普比率计算"""
    annual_return = np.dot(weights, mean_returns) * 252
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    # 权重越界惩罚
    penalty = np.sum(np.where(weights < MIN_WEIGHT, (MIN_WEIGHT - weights) * 100, 0))
    penalty += np.sum(np.where(weights > MAX_WEIGHT, (weights - MAX_WEIGHT) * 100, 0))

    excess_return = annual_return - RISK_FREE_RATE
    if annual_volatility == 0:
        return -np.inf, annual_return, annual_volatility
    return (excess_return / annual_volatility) - penalty, annual_return, annual_volatility


# ===== 增强的WOA算法 =====
def enhanced_whale_optimization(mean_returns, cov_matrix, n_assets):
    population = initialize_population(n_assets)
    fitness = [calculate_sharpe(ind, mean_returns, cov_matrix)[0] for ind in population]
    best_idx = np.argmax(fitness)
    best_weights = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = []
    perturbation_rate = 0.3  # 初始扰动率

    for iter in range(MAX_ITER):
        a = 2.0 * np.exp(-5.0 * iter / MAX_ITER)
        a2 = -1.0 + iter / MAX_ITER

        # 修正后的扰动率计算
        perturbation_rate = 0.3 * (1 - iter / MAX_ITER)**2

        for i in range(POP_SIZE):
            if np.random.rand() < 0.1:
                new_solution = best_weights.copy()
            else:
                A = 2.0 * a * np.random.rand() - a
                C = 2.0 * np.random.rand()
                p = np.random.rand()
                b = 1.0 + 0.5 * np.sin(np.pi * iter / MAX_ITER)

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * best_weights - population[i])
                        new_solution = best_weights - A * D
                    else:
                        rand_idx = np.random.randint(POP_SIZE)
                        D = abs(C * population[rand_idx] - population[i])
                        new_solution = population[rand_idx] - A * D
                else:
                    D = abs(best_weights - population[i])
                    l = np.random.uniform(-1, 1)
                    new_solution = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_weights

            # 动态高斯扰动
            if np.random.rand() < perturbation_rate:
                new_solution += np.random.normal(0, 0.1 * (1 - iter / MAX_ITER), n_assets)

            # 自适应变异
            if iter > MAX_ITER // 2 and np.random.rand() < 0.2:
                mut_idx = np.random.randint(n_assets)
                new_solution[mut_idx] = np.random.uniform(MIN_WEIGHT, MAX_WEIGHT)

            new_solution = repair_weights(new_solution, MIN_WEIGHT, MAX_WEIGHT)
            new_fitness = calculate_sharpe(new_solution, mean_returns, cov_matrix)[0]

            # 模拟退火接受准则
            if new_fitness > fitness[i] or np.random.rand() < np.exp((new_fitness - fitness[i]) / max(a, 1e-6)):
                population[i] = new_solution
                fitness[i] = new_fitness
                if new_fitness > best_fitness:
                    best_weights = new_solution.copy()
                    best_fitness = new_fitness

        # 局部搜索阶段
        if iter % 50 == 0 and iter > MAX_ITER // 3:
            local_search_weights = best_weights + np.random.normal(0, 0.02, n_assets)
            local_search_weights = repair_weights(local_search_weights, MIN_WEIGHT, MAX_WEIGHT)
            local_fitness = calculate_sharpe(local_search_weights, mean_returns, cov_matrix)[0]
            if local_fitness > best_fitness:
                best_weights = local_search_weights.copy()
                best_fitness = local_fitness

        convergence_curve.append(best_fitness)

    return best_weights, convergence_curve


# ===== 可视化增强 =====
def plot_convergence(convergence, window=10):
    """绘制收敛曲线（含移动平均）"""
    plt.figure(figsize=(10, 6))
    ma = pd.Series(convergence).rolling(window).mean()

    plt.plot(convergence, label='原始值', alpha=0.3)
    plt.plot(ma, label=f'{window}期移动平均', color='red')
    plt.title("优化过程收敛曲线", fontsize=14)
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel("年化夏普比率", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, "收敛曲线.png"))
    plt.close()


# ===== 主函数 =====
def optimize_portfolio(filepath):
    # 数据准备
    df = pd.read_excel(filepath)
    returns = df.iloc[:, 1:].pct_change().dropna().values
    stock_names = df.columns[1:]

    # 计算统计量
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)

    # 执行优化
    weights, convergence = enhanced_whale_optimization(mean_returns, cov_matrix, len(stock_names))
    sharpe, ret, risk = calculate_sharpe(weights, mean_returns, cov_matrix)

    # 保存结果
    result_df = pd.DataFrame({
        '资产': stock_names,
        '权重': weights,
        '是否达标': ['是' if MIN_WEIGHT <= w <= MAX_WEIGHT else '否' for w in weights]
    })
    result_df.to_excel(os.path.join(RESULT_DIR, "最优权重.xlsx"), index=False)

    # 保存指标
    pd.DataFrame({
        '年化收益率': [ret],
        '年化波动率': [risk],
        '夏普比率': [sharpe]
    }).to_excel(os.path.join(RESULT_DIR, "组合指标.xlsx"), index=False)

    # 绘制图形
    plot_convergence(convergence)

    # 绘制权重分布
    plt.figure(figsize=(10, 10))
    plt.pie(weights, labels=stock_names, autopct='%1.1f%%', startangle=90)
    plt.title('资产配置权重分布')
    plt.savefig(os.path.join(RESULT_DIR, "权重分布.png"))
    plt.close()

    print("优化完成！结果已保存至目录:", RESULT_DIR)


# ===== 执行优化 =====
if __name__ == "__main__":
    optimize_portfolio("预测回报率.xlsx")

