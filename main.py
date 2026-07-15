
# src/solver.py

def zero_one_knapsack_2d(products, budget):
    """
    二维0/1背包动态规划算法，返回最大价值和所选商品ID列表
    :param products: 列表，元素为元组 (商品ID, 价值, 成本)
    :param budget: 总预算（背包容量）
    :return: 最大价值, 所选商品ID列表
    """
    n = len(products)
    # 初始化DP表：dp[i][j] 表示前i个商品、预算j时的最大价值
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    # 填充DP表
    for i in range(1, n + 1):
        prod_id, value, cost = products[i - 1]
        for j in range(1, budget + 1):
            if cost > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - cost] + value)

    # 回溯查找所选商品
    selected_ids = []
    j = budget
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            selected_ids.append(products[i - 1][0])
            j -= products[i - 1][2]

    return dp[n][budget], selected_ids[::-1]  # 逆序后返回所选商品ID


def zero_one_knapsack_1d(products, budget):
    """
    一维滚动数组0/1背包动态规划算法（空间优化版），返回最大价值和所选商品ID列表
    """
    dp = [0] * (budget + 1)
    # 记录每个预算下的选择（用于回溯）
    selected = [[False] * (budget + 1) for _ in range(len(products) + 1)]

    for i in range(len(products)):
        prod_id, value, cost = products[i]
        # 逆序遍历预算，避免重复选择同一商品
        for j in range(budget, cost - 1, -1):
            if dp[j - cost] + value > dp[j]:
                dp[j] = dp[j - cost] + value
                selected[i + 1][j] = True  # 标记第i+1个商品在预算j时被选中

    # 回溯查找所选商品
    selected_ids = []
    j = budget
    for i in range(len(products), 0, -1):
        if selected[i][j]:
            selected_ids.append(products[i - 1][0])
            j -= products[i - 1][2]

    return dp[budget], selected_ids[::-1]


if __name__ == "__main__":
    # 测试用例：3个商品，预算10
    test_products = [
        (1, 5, 3),  # 商品1：价值5，成本3
        (2, 8, 5),  # 商品2：价值8，成本5
        (3, 10, 6)  # 商品3：价值10，成本6
    ]
    test_budget = 10
    expected_value = 15  # 预期最大价值：商品1+商品3 → 5+10=15
    expected_ids = [1, 3]

    # 测试二维DP
    value_2d, ids_2d = zero_one_knapsack_2d(test_products, test_budget)
    print("=== 二维DP测试结果 ===")
    print(f"最大价值：{value_2d}，所选商品：{ids_2d}")
    assert value_2d == expected_value, "二维DP价值计算错误"
    assert ids_2d == expected_ids, "二维DP商品选择错误"
    print("二维DP测试通过！\n")

    # 测试一维DP
    value_1d, ids_1d = zero_one_knapsack_1d(test_products, test_budget)
    print("=== 一维DP测试结果 ===")
    print(f"最大价值：{value_1d}，所选商品：{ids_1d}")
    assert value_1d == expected_value, "一维DP价值计算错误"
    assert ids_1d == expected_ids, "一维DP商品选择错误"
    print("一维DP测试通过！")
