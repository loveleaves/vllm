## sleep mode
- 初始vllm代码v0.6.6
- 当前仓库commit：v0.6.6-fix
- [安装说明](./docs/sleep_mode/installation.md)
- [vllm设计说明](./docs/sleep_mode/sleep_mode_internals.md)
- [ascend设计](./docs/sleep_mode/sleep_mode_design_ascend.md)
- [测试](./tests/sleep_mode)

## problems
### network
```
[Environment]::SetEnvironmentVariable("HTTPS_PROXY", "http://127.0.0.1:6789", "User")
[Environment]::SetEnvironmentVariable("HTTP_PROXY", "http://127.0.0.1:6789", "User")
```