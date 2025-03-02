import numpy as np
from scipy.optimize import minimize
from numba import njit
import multiprocessing

def loss(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true))

# 目标函数：计算不同模型的误差
def objective(params, x, y, model_func):
    y_pred = model_func(x, *params)
    return loss(y, y_pred)

# 并行拟合单个模型
def fit_model(params_initial_guess, x, y, model_func):
    result = minimize(objective, params_initial_guess, args=(x, y, model_func),
                      method='L-BFGS-B', options={'maxiter': 10})
    return result.fun, result.x 

# 并行处理数据拟合（每个模型独立计算）
def fit_data_parallel(x, ys):
    initial_guess = [0.1, 0.1, 0.1, 0.1]
    models = [model1, model2, model3]
    
    error_map = np.zeros((len(ys), len(models)))  # 误差表 (10000, model_num)
    param_map = np.zeros((len(ys), len(models), 4))  # 存储每个模型的参数
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(fit_model, [(initial_guess, x, y, model) for model in models for y in ys])
    
    # 填充误差表和参数表
    for i in range(len(ys)):
        for j in range(len(models)):
            error_map[i, j], param_map[i, j] = results[i * len(models) + j]
    
    # 选择误差最小的模型
    best_model_indices = np.argmin(error_map, axis=1)  # 获取误差最小的模型索引
    best_results = [(models[idx].__name__, param_map[i, idx]) for i, idx in enumerate(best_model_indices)]
    
    return best_results

if __name__ == '__main__':
    import time
    start_time = time.time()
    x = np.array([0.1, 0.32, 0.48, 0.87], dtype=np.float32)
    ys = np.random.random_sample((10000, 4)).astype(np.float32)
    
    fitted_results = fit_data_parallel(x, ys)
    for i, (model_name, params) in enumerate(fitted_results[:5]):
        print(f"Best model for y{i+1}: {model_name} with parameters: {params}")
    
    end_time = time.time()
    print('Finished, cost:', end_time - start_time)
