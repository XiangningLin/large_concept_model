# Globus Connection Module

在 VastAI 实例上还原预打包的 Globus Connect Personal (GCP) 凭证，并将预处理数据自动传输到 NCSA Delta。

**流程**：本地一次性准备（含人工步骤）→ VastAI onstart 全自动零人工。

## 功能

1. **install_gcp.sh**：还原预打包凭证、启动 GCP
2. **transfer.py**：使用 Globus Transfer API 将数据传到 Delta
3. **local_prep/**：本地准备脚本（获取 refresh token、打包 GCP 凭证）

## 环境变量

**VastAI 环境变量上限 16 个**，长字符串（如凭证 base64）需写在 `lcm_preprocess_onstart.sh` 的 `GLOBUS_CREDS` 中，不占 env 配额。

### GCP 安装（install_gcp.sh）


| 变量                       | 必需  | 说明                                          |
| ------------------------ | --- | ------------------------------------------- |
| `GLOBUS_CREDS_B64`       | 是*  | 凭证包 base64；由脚本从 `GLOBUS_CREDS` 导出，或由 VastAI env 传入 |
| `GLOBUS_GCP_INSTALL_DIR` | 否   | 安装目录，默认 `/workspace/globus_gcp`             |
| `GLOBUS_SOURCE_PATH`     | 否   | 源数据路径，默认 `/workspace/lcm/preprocessed_data` |

### 传输认证（transfer.py）


| 变量                           | 必需  | 说明                                                   |
| ---------------------------- | --- | ---------------------------------------------------- |
| `GLOBUS_REFRESH_TOKEN`       | 是   | OAuth refresh token                                  |
| `GLOBUS_CLIENT_ID`           | 是   | Native App 的 client_id                               |
| `GLOBUS_CLIENT_SECRET`       | 否   | 若 Native 刷新失败，改用 Confidential App 时填写                |
| `GLOBUS_SOURCE_ENDPOINT`     | 否   | 源 endpoint UUID，不设则从 GCP 自动检测                        |
| `GLOBUS_DEST_ENDPOINT`       | 否   | 目标 endpoint，默认 NCSA Delta                            |
| `GLOBUS_DEST_PATH`           | 否   | 目标路径，默认 `/work/hdd/bfaq/jlyu3/lcm/preprocessed_data` |
| `GLOBUS_TRANSFER_LABEL`      | 否   | 传输任务标签                                               |
| `GLOBUS_WAIT_FOR_COMPLETION` | 否   | `true` 时等待传输完成                                       |
| `GLOBUS_SYNC_LEVEL`          | 否   | 同步级别，默认 `checksum`                                   |


## 使用流程

### 第一部分：本地准备（一次性，含人工）

详见 [local_prep/README.md](local_prep/README.md)，简要步骤：

1. 在 developers.globus.org 创建 Native App，记下 client_id
2. 运行 `get_refresh_token.py`，浏览器授权，获取 refresh_token
3. 本地完成 GCP setup（浏览器授权），运行 `prepare_gcp_creds.sh` 打包凭证
4. 将 `globus_creds.b64` 内容填入 `lcm_preprocess_onstart.sh` 的 `GLOBUS_CREDS`；refresh_token、client_id 存入 VastAI 环境变量（最多 16 个）

### 第二部分：VastAI onstart（全自动）

配置好环境变量后，每次实例启动：

1. 还原 GCP 凭证并启动
2. 运行 LCM 预处理
3. 预处理完成后自动提交 Globus 传输到 Delta

无需任何人工操作。

## 依赖

- `globus-sdk`（transfer.py 需要）：`pip install globus-sdk`

## 安全

- 不要将 `GLOBUS_REFRESH_TOKEN`、`GLOBUS_CREDS_B64` 等写入代码或公开仓库
- 使用 VastAI 的 Environment Variables 或 secrets 管理

## Delta Endpoint

NCSA Delta 的 Globus endpoint UUID 可在 [Globus Endpoint Search](https://app.globus.org/endpoints) 或 NCSA 文档中查询。默认使用 `82f1b5c6-6e9b-11e9-bf45-0e4a062367b1`，如有变更请设置 `GLOBUS_DEST_ENDPOINT`。

## 兼容性

- 未设置 `GLOBUS_CREDS_B64`（脚本内 `GLOBUS_CREDS` 或 VastAI env）时，Globus 相关逻辑全部跳过，onstart 行为与未配置时一致

