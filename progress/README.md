# 进度与状态语义

`./lab status` 列出所有 27 关的状态和原因，不运行学生 kernel；`--stage N` 可限定阶段。完成一关的实现后设置其 `IMPLEMENTED=True`，再运行 `./lab check ID` 验收，或 `./lab bench ID` 先校验再计时。

- `INFRASTRUCTURE_OK`：smoke 的 import/构建/参考/标量计时设施正常，不是学生正确性。
- `STUDENT_UNIMPLEMENTED`：学生 flag 未开启、核心 stub 报错或命名的复用模块缺失。
- `IMPLEMENTED_UNVERIFIED`：已开启但没有当前源码的完整验收，或旧记录已过期。
- `CORRECTNESS_PASSED`：所有声明 cases 的数值/重复/stream 检查通过；不意味着机制设计审查或 sanitizer 自动通过。
- `CORRECTNESS_FAILED`：当前实现的验收或执行失败，附原因；基础设施 bug 应独立定位。
- `ENV_UNSUPPORTED`：所选 GPU/编译器/包缺失。工具缺失和权限受限也明确报错，不算通过。

`progress/results/ID.json` 是可再生成的机器记录，包含代码指纹。代码指纹覆盖所有学生源码与公共设施（保守失效策略），所以前关 helper 变动会使旧结果失效。单 case 成功不能覆盖为全关通过。`experiments/ID.md` 保存人工推导、性能、sanitizer、设计 review 与下一步；这些笔记不能自动覆盖。

终端使用英文状态：NOT IMPLEMENTED / UNVERIFIED / PASS / FAIL / UNSUPPORTED。`--json` 保留上面的完整状态名；`--verbose` 展开详细原因和报告路径。

退出码：0 完成该命令；2 学生未实现；3 数值/断言或 sanitizer 发现错误；4 环境不支持；5 超时；6 参数/编译/运行错误。默认 pytest 仅基础设施；显式 `pytest tests/student` 会对未实现关失败，绝不 skip/xfail。
