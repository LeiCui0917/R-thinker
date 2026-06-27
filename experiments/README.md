# Think_Agent 实验套件

本目录包含 Think_Agent（快慢双系统）相关实验入口脚本与结果输出。

## 实验脚本（按编号）
- `run_exp1_main_baselines.py`：主基线对比（Think_Agent vs Reflexion/CoT/LLM_Agent/MemoryLLM/Coding/Rule1）。
- `run_exp2_ablation.py`：消融实验（FastOnly/SlowOnly/wo_opp/wo_self/forward）。
- `run_exp3_round_robin.py`：Exp1 智能体池 round-robin 两两对抗。
- `run_exp4_rule_delay.py`：Rule1 延迟敏感性实验（不同 delay）。
- `run_exp5_time_budget.py`：单步决策时间预算实验（不同 budget）。
- `run_exp6_env_sweep.py`：环境参数扫面（临时改 `config.json` 后调用实验 1）。

运行示例：
```bash
python Experiments/run_exp1_main_baselines.py
python Experiments/run_exp2_ablation.py
python Experiments/run_exp5_time_budget.py
```

输出位置：`Experiments/results/` 下生成对应 CSV。

## 智能体说明（简要）
- Think_Agent（Ours）: 使用 ThinkAgent，将快（Fast）与慢（Slow）模块协同；慢模块含自我/对手双视角树与指导。
- Reflexion/CoT: 基于 LLM baseline 的过程增强策略（分别强调持续反思与自一致性）。
- MemoryLLM: 单 LLM，维护简单记忆，将历史片段附加到提示词前端。
- Coding: 单 LLM 生成“状态-动作对（StateAction）”清单与执行要点，并通过规则模块选择可执行动作。
- FastOnly: 单快模块（等价于原 LLMAgent）。
- SlowOnly: 慢模块直出动作并解析（Maze 方向字母；Chess UCI）。
- Think_Agent_wo_opp / Think_Agent_wo_self: 分别移除敌树指导或我树指导的变体。

详细请见 `agents/` 中各说明文件。
