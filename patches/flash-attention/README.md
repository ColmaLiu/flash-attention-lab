# FA4 rescale 配置补丁

`rescale-threshold.patch` 将 SM100 forward 的 16-bit softmax rescale 阈值暴露为构造参数，默认值为上游使用的 8.0。课程 s40–s43 可以因此独立比较阈值 0 与 8，并保持公共 O/LSE 契约。

上游版本由 `third_party/flash-attention` 的 gitlink 固定。`flash_attention_lab/source_cache.py` 校验版本，将 CuTe 源码复制到 `.cache/sources/flash-attention/`，检查并应用补丁。缓存键包含源码和补丁内容，构建副本不会修改 submodule；修改补丁后生成新的副本。

原始代码、作者与许可在 submodule 内。阅读入口及实验配置见 [参考实现](../../docs/reference-implementations.md)。
