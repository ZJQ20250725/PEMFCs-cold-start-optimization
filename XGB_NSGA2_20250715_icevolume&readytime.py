"""
Created on 2025.07.15

@Source code author: Zhu, J.Q

This system is a multi-objective optimization tool for fuel cell operating parameters based on the improved NSGA-2 (Non-dominated Sorting Genetic Algorithm II).
By combining the pre-trained XGBoost model and SHAP (Shapley Additive Explanations) feature importance analysis, it achieves intelligent optimization of the key operating parameters of the fuel cell,
aiming to simultaneously minimize the preheating time (Ready Time) and the ice volume fraction (Ice Volume) as the two core objectives.

"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, qmc
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import joblib
import os
import logging
import matplotlib.font_manager as fm
import shap  # 新增SHAP库依赖

# 配置日志记录
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class FuelCellOptimizer:
    def __init__(self, random_seed=42, initial_weights=(-1.0, -1.0)):
        # 设置随机种子
        self.random_seed = random_seed
        self._set_random_seeds()

        # 模型路径和特征名称
        self.xgb_readytime_path = 'readytime_XGB_optimized.pkl'
        self.xgb_icevolume_path = 'icevolume_XGB_optimized.pkl'
        self.feature_names = [
            "Dry_molar_flow_rate", "Current_set", "Pressure",
            "Pump_stoichiometry", "Relative_humidity_H2",
            "Relative_humidity_air", "Temperature_air", "Temperature_hydrogen"
        ]

        # 特征参数范围
        self.feature_ranges = {
            "Dry_molar_flow_rate": (0.05, 4.0),
            "Current_set": (20, 100),
            "Pressure": (140, 260),
            "Pump_stoichiometry": (0.02, 0.15),
            "Relative_humidity_H2": (30, 100),
            "Relative_humidity_air": (30, 100),
            "Temperature_air": (253, 343),
            "Temperature_hydrogen": (253, 343)
        }

        # 目标值实际范围
        self.readytime_range = (21, 115)
        self.icevolume_range = (0.3, 33)

        # 初始化模型和优化参数
        self.xgb_readytime = None
        self.xgb_icevolume = None
        self.toolbox = None
        self.population = None
        self.hof = None
        self.stats = None

        # 优化结果
        self.pareto_front_individuals = None
        self.pareto_front_y1 = None  # 预热时间
        self.pareto_front_y2 = None  # 工作时间

        # 初始权重（固定）
        self.initial_weights = initial_weights

        # 约束与多样性监控
        self.constraint_violations = 0
        self.diversity_history = []  # 记录种群多样性
        self.convergence_threshold = 1e-4  # 收敛判断阈值
        self.convergence_patience = 15  # 连续收敛代数阈值

        # 评估缓存，避免重复计算
        self.evaluation_cache = {}

        # 新增SHAP相关属性
        self.shap_weights = None  # 存储各特征的SHAP权重
        self.shap_background_samples = 200  # 用于计算SHAP的背景样本数量

    def _set_random_seeds(self):
        random.seed(int(self.random_seed))
        np.random.seed(int(self.random_seed))

    def load_models(self):
        """加载预训练模型"""
        model_files = [self.xgb_readytime_path, self.xgb_icevolume_path]
        for file in model_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Error: {file} does not exist.")

        try:
            self.xgb_readytime = joblib.load(self.xgb_readytime_path)
            print("readytime_XGB_optimized model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ready time model: {e}")
            raise

        try:
            self.xgb_icevolume = joblib.load(self.xgb_icevolume_path)
            print("icevolume_XGB_optimized model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load work time model: {e}")
            raise

    def calculate_shap_importance(self):
        """计算各特征对两个目标的平均SHAP贡献值作为权重"""
        if self.xgb_readytime is None or self.xgb_icevolume is None:
            raise ValueError("请先加载模型再计算SHAP值")

        # 1. 生成背景样本（覆盖特征空间）
        sampler = qmc.LatinHypercube(d=len(self.feature_names), seed=self.random_seed)
        background_norm = sampler.random(n=self.shap_background_samples)  # 归一化样本
        # 转换为实际特征值
        background_actual = np.array([
            [self.denormalize_feature(x, self.feature_names[i])
             for i, x in enumerate(sample)]
            for sample in background_norm
        ])

        # 2. 为两个模型计算SHAP值
        explainer_ready = shap.TreeExplainer(self.xgb_readytime)
        shap_values_ready = explainer_ready.shap_values(background_actual)

        explainer_work = shap.TreeExplainer(self.xgb_icevolume)
        shap_values_work = explainer_work.shap_values(background_actual)

        # 3. 计算特征的平均SHAP贡献（取绝对值，因为重要性与正负无关）
        avg_shap_ready = np.mean(np.abs(shap_values_ready), axis=0)
        avg_shap_work = np.mean(np.abs(shap_values_work), axis=0)

        # 4. 合并两个目标的SHAP值，取平均作为最终权重
        combined_shap = (avg_shap_ready + avg_shap_work) / 2

        # 5. 归一化权重（使权重之和为1，便于缩放）
        self.shap_weights = combined_shap / np.sum(combined_shap)

        print("特征SHAP权重计算完成:")
        for name, weight in zip(self.feature_names, self.shap_weights):
            print(f"  {name}: {weight:.4f}")

    def normalize_feature(self, value, feature_name):
        """将特征值归一化到[0,1]范围内"""
        if feature_name not in self.feature_ranges:
            raise ValueError(f"Feature {feature_name} not found in ranges.")

        min_val, max_val = self.feature_ranges[feature_name]
        if max_val == min_val:
            return 0.5  # 中间值

        return (value - min_val) / (max_val - min_val)

    def denormalize_feature(self, norm_value, feature_name):
        """将[0,1]范围内的归一化值反归一化为实际特征值"""
        if feature_name not in self.feature_ranges:
            raise ValueError(f"Feature {feature_name} not found in ranges.")

        min_val, max_val = self.feature_ranges[feature_name]
        return min_val + norm_value * (max_val - min_val)

    def denormalize_readytime(self, norm_value):
        """反归一化预热时间"""
        min_val, max_val = self.readytime_range
        return min_val + norm_value * (max_val - min_val)

    def denormalize_icevolume(self, norm_value):
        """反归一化冰体积分数"""
        min_val, max_val = self.icevolume_range
        return min_val + norm_value * (max_val - min_val)

    def check_bounds(self, individual):
        """确保个体的每个特征值都在[0,1]归一化范围内"""
        for i in range(len(individual)):
            if individual[i] < 0.0:
                individual[i] = 0.0
                self.constraint_violations += 1
            elif individual[i] > 1.0:
                individual[i] = 1.0
                self.constraint_violations += 1
        return individual

    def create_individual(self):
        """使用拉丁超立方抽样创建初始个体，提高分布均匀性"""
        sampler = qmc.LatinHypercube(d=len(self.feature_names), seed=self.random_seed)
        sample = sampler.random(n=1)[0]  # 生成[0,1)范围内的样本
        return creator.Individual(sample)

    def evaluate(self, individual):
        """评估函数 - 预测预热时间和工作时间"""
        if len(individual) != len(self.feature_names):
            raise ValueError(f"Individual must have {len(self.feature_names)} features.")

        # 检查边界并修复
        individual = self.check_bounds(individual)

        # 检查缓存，避免重复评估
        ind_tuple = tuple(round(x, 6) for x in individual)  # 四舍五入减少缓存键数量
        if ind_tuple in self.evaluation_cache:
            return self.evaluation_cache[ind_tuple]

        # 将归一化的个体转换为实际特征值
        actual_features = [
            self.denormalize_feature(individual[i], self.feature_names[i])
            for i in range(len(self.feature_names))
        ]

        X_test = np.array(actual_features).reshape(1, -1)

        try:
            y1_pred_norm = self.xgb_readytime.predict(X_test).flatten()[0]
            y2_pred_norm = self.xgb_icevolume.predict(X_test).flatten()[0]
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return float('inf'), float('inf')

        # 反归一化目标值
        y1_pred = self.denormalize_readytime(y1_pred_norm)
        y2_pred = self.denormalize_icevolume(y2_pred_norm)

        # 添加约束惩罚 - 目标值低于下限时的惩罚
        penalty = 0.0
        if y1_pred < self.readytime_range[0]:
            penalty += (self.readytime_range[0] - y1_pred) * 10  # 惩罚系数
        if y2_pred < self.icevolume_range[0]:
            penalty += (self.icevolume_range[0] - y2_pred) * 10  # 惩罚系数

        # 缓存结果
        result = (y1_pred + penalty, y2_pred + penalty)
        self.evaluation_cache[ind_tuple] = result
        return result

    def initialize_toolbox(self):
        """初始化DEAP工具箱"""
        creator.create("FitnessMulti", base.Fitness, weights=self.initial_weights)
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.custom_mutate)  # 注册修改后的变异方法
        self.toolbox.register("select", tools.selNSGA2)

    def custom_mutate(self, individual, base_sigma=0.1):
        """基于SHAP贡献值的差异化变异算子
        特征扰动幅度 = 基础sigma * (1 + SHAP权重放大系数)
        确保贡献高的特征有更大扰动
        """
        if self.shap_weights is None:
            raise ValueError("请先调用calculate_shap_importance计算SHAP权重")

        # 计算权重放大系数（将SHAP权重映射到[0, 1]范围）
        max_weight = np.max(self.shap_weights)
        weight_scaler = self.shap_weights / max_weight  # 最大权重对应scaler=1

        for i in range(len(individual)):
            # 为每个特征计算差异化sigma
            feature_sigma = base_sigma * (1 + weight_scaler[i])
            # 应用高斯变异
            individual[i] += random.gauss(0, feature_sigma)

        return individual,

    def calculate_diversity(self, population):
        """计算种群多样性（个体间平均欧氏距离）"""
        if len(population) < 2:
            return 0.0
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
                distances.append(dist)
        return np.mean(distances)

    def run_optimization(self, pop_size=200, num_gens=100,
                         initial_cxpb=0.9, final_cxpb=0.4,
                         initial_mutpb=0.8, final_mutpb=0.1,
                         initial_sigma=0.2, final_sigma=0.05):
        """运行NSGA-2优化算法（带精英保留和自适应算子）"""
        if self.toolbox is None:
            self.initialize_toolbox()

        self.constraint_violations = 0
        self.diversity_history = []
        self.evaluation_cache = {}  # 重置缓存
        self.population = self.toolbox.population(n=pop_size)
        self.hof = tools.ParetoFront()

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        # 评估初始种群
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 收敛监控变量
        convergence_count = 0
        prev_diversity = None

        # 主循环
        for gen in range(num_gens):
            # 自适应调整交叉和变异概率
            cxpb = initial_cxpb - (initial_cxpb - final_cxpb) * (gen / num_gens)
            mutpb = initial_mutpb - (initial_mutpb - final_mutpb) * (gen / num_gens)
            base_sigma = initial_sigma - (initial_sigma - final_sigma) * (gen / num_gens)  # 基础变异强度

            # 生成后代
            offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=cxpb, mutpb=mutpb)

            # 应用当前代的差异化变异
            for ind in offspring:
                if random.random() < mutpb:
                    self.custom_mutate(ind, base_sigma=base_sigma)  # 使用基础sigma计算差异化扰动
                self.check_bounds(ind)

            # 评估后代
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 精英保留：合并前代和后代并选择
            combined = self.population + offspring
            self.population = self.toolbox.select(combined, k=len(self.population))

            # 更新Hall of Fame
            self.hof.update(combined)

            # 计算并记录种群多样性
            diversity = self.calculate_diversity(self.population)
            self.diversity_history.append(diversity)

            # 每10代打印一次信息
            if gen % 10 == 0 or gen == num_gens - 1:
                print(
                    f"Generation {gen}: Diversity = {diversity:.6f}, CXPB = {cxpb:.3f}, MUTPB = {mutpb:.3f}, Base Sigma = {base_sigma:.3f}")

            # 收敛判断
            if prev_diversity is not None:
                diversity_change = abs(diversity - prev_diversity)
                if diversity_change < self.convergence_threshold:
                    convergence_count += 1
                    if convergence_count >= self.convergence_patience:
                        print(f"Early stopping at generation {gen} due to convergence.")
                        break
                else:
                    convergence_count = 0
            prev_diversity = diversity

        # 保存优化结果
        self.pareto_front_individuals = self.hof.items
        self.pareto_front_y1 = [ind.fitness.values[0] for ind in self.pareto_front_individuals]
        self.pareto_front_y2 = [ind.fitness.values[1] for ind in self.pareto_front_individuals]

        print(f"Total constraint violations: {self.constraint_violations}")
        print(f"Total unique evaluations: {len(self.evaluation_cache)}")
        self.visualize_diversity()

    def visualize_diversity(self, filename='population_diversity.png'):
        """可视化种群多样性变化"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.diversity_history)), self.diversity_history, 'b-', linewidth=2)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Population Diversity (Average Euclidean Distance)', fontsize=14)
        plt.title('Population Diversity Over Generations', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    def _get_feature_data(self, ind):
        norm_features = [round(x, 4) for x in ind]
        actual_features = [self.denormalize_feature(x, self.feature_names[j])
                           for j, x in enumerate(ind)]
        actual_features = [round(x, 4) for x in actual_features]
        return norm_features, actual_features

    def save_results_to_csv(self, filename='pareto_front.csv'):
        if self.pareto_front_individuals is None:
            print("No optimization results to save. Run optimization first.")
            return

        data = {
            'Dry_molar_flow_rate': [],
            'Flowsheet_current_set': [],
            'Pressure': [],
            'Pump_stoichiometry': [],
            'Relative_humidity_H2': [],
            'Relative_humidity_air': [],
            'Temperature_air': [],
            'Temperature_hydrogen': [],
            'Normalized_Dry_molar_flow_rate': [],
            'Normalized_Flowsheet_current_set': [],
            'Normalized_Pressure': [],
            'Normalized_Pump_stoichiometry': [],
            'Normalized_Relative_humidity_H2': [],
            'Normalized_Relative_humidity_air': [],
            'Normalized_Temperature_air': [],
            'Normalized_Temperature_hydrogen': [],
            'ReadyTime': self.pareto_front_y1,
            'Icevolume': self.pareto_front_y2
        }

        for ind in self.pareto_front_individuals:
            norm_features, actual_features = self._get_feature_data(ind)
            data['Dry_molar_flow_rate'].append(actual_features[0])
            data['Flowsheet_current_set'].append(actual_features[1])
            data['Pressure'].append(actual_features[2])
            data['Pump_stoichiometry'].append(actual_features[3])
            data['Relative_humidity_H2'].append(actual_features[4])
            data['Relative_humidity_air'].append(actual_features[5])
            data['Temperature_air'].append(actual_features[6])
            data['Temperature_hydrogen'].append(actual_features[7])
            data['Normalized_Dry_molar_flow_rate'].append(norm_features[0])
            data['Normalized_Flowsheet_current_set'].append(norm_features[1])
            data['Normalized_Pressure'].append(norm_features[2])
            data['Normalized_Pump_stoichiometry'].append(norm_features[3])
            data['Normalized_Relative_humidity_H2'].append(norm_features[4])
            data['Normalized_Relative_humidity_air'].append(norm_features[5])
            data['Normalized_Temperature_air'].append(norm_features[6])
            data['Normalized_Temperature_hydrogen'].append(norm_features[7])

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\nPareto front data saved to '{filename}'")

    def visualize_pareto_front(self, filename='pareto_front.png'):
        if self.pareto_front_individuals is None or not self.pareto_front_y1 or not self.pareto_front_y2:
            print("No optimization results to visualize. Run optimization first.")
            return

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'

        correlation, _ = pearsonr(self.pareto_front_y1, self.pareto_front_y2)

        plt.figure(figsize=(8, 6))
        plt.scatter(self.pareto_front_y1, self.pareto_front_y2,
                    color='red', alpha=0.7, s=50)

        # 绘制目标范围下限
        plt.axvline(x=self.readytime_range[0], color='blue', linestyle='--', alpha=0.5, label='ReadyTime下限')
        plt.axhline(y=self.icevolume_range[0], color='green', linestyle='--', alpha=0.5, label='WorkTime下限')

        plt.xlabel('Ready Time (s)', fontsize=26, labelpad=10)
        plt.ylabel('Ice Volume (vol%)', fontsize=26, labelpad=10)
        plt.title('Pareto Optimal Front', fontsize=26, pad=20)

        plt.text(0.05, 0.95, f'Correlation = {correlation:.2f}',
                 transform=plt.gca().transAxes,
                 fontsize=22, verticalalignment='top')

        plt.legend()
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Pareto front visualization saved to '{filename}'")

    def visualize_decision_variables(self, filename='decision_variables.png'):
        if self.pareto_front_individuals is None:
            print("No optimization results to visualize. Run optimization first.")
            return

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'

        plt.figure(figsize=(12, 8))

        feature_data = []
        for ind in self.pareto_front_individuals:
            if len(ind) >= 8:
                feature_data.append(ind[:8])

        if not feature_data:
            print("No valid decision variables to plot.")
            return

        feature_data = list(map(list, zip(*feature_data)))

        bplot = plt.boxplot(feature_data,
                            patch_artist=True,
                            labels=self.feature_names[:len(feature_data)])

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.title('Decision Variables Distribution (Normalized Values)', fontsize=26, pad=20)
        plt.ylabel('Normalized Value', fontsize=26, labelpad=10)

        plt.xticks(rotation=45, ha='right', fontsize=22)
        plt.yticks(fontsize=22)

        # 绘制[0,1]范围参考线
        plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Decision variables visualization saved to '{filename}'")

    def print_pareto_solutions(self):
        if self.pareto_front_individuals is None:
            print("No optimization results to print. Run optimization first.")
            return

        print("\nPareto Front Solutions:")
        for i, ind in enumerate(self.pareto_front_individuals[:5]):  # 只打印前5个解
            norm_features, actual_features = self._get_feature_data(ind)
            print(f"Solution {i + 1}:")
            print(f"  Ready Time: {ind.fitness.values[0]:.4f} seconds, Ice Volume: {ind.fitness.values[1]:.4f} seconds")
            print(f"  Actual Features: {actual_features}")


def main():
    optimizer = FuelCellOptimizer(random_seed=42, initial_weights=(-0.5, -0.05))

    try:
        optimizer.load_models()
        optimizer.calculate_shap_importance()  # 计算SHAP权重

        # 打印扩大后的特征范围
        print("\nExpanded Feature Ranges:")
        for feature in optimizer.feature_names:
            min_val, max_val = optimizer.feature_ranges[feature]
            print(f"{feature}: [{min_val:.3f}, {max_val:.3f}]")

        # 运行优化（带增强的自适应参数）
        optimizer.run_optimization(
            pop_size=200,
            num_gens=100,
            initial_cxpb=0.95,
            final_cxpb=0.4,
            initial_mutpb=0.85,
            final_mutpb=0.4,
            initial_sigma=0.25,  # 初始基础变异强度
            final_sigma=0.08  # 最终基础变异强度
        )

        optimizer.print_pareto_solutions()
        optimizer.save_results_to_csv()
        optimizer.visualize_pareto_front()
        optimizer.visualize_decision_variables()

    except Exception as e:
        print(f"An error occurred during optimization: {e}")


if __name__ == "__main__":
    main()