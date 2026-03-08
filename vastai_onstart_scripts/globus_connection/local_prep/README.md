# Globus 本地准备脚本

在本地执行一次，获取 VastAI onstart 全自动传输所需的凭证。完成后，每次 VastAI 实例启动无需任何人工操作。

## 步骤 1：注册 Globus App

1. 登录 [developers.globus.org](https://developers.globus.org/)
2. 创建 **Native App**（不是 Confidential App）
3. Redirect URI 填：`https://auth.globus.org/v2/web/auth-code`
4. 记下 **client_id**（UUID）

## 步骤 2：获取 refresh token

```bash
pip install globus-sdk
cd large_concept_model/vastai_onstart_scripts/globus_connection/local_prep
python get_refresh_token.py <你的client_id>
```

- 浏览器打开打印的 URL，登录 Globus 并授权
- 将授权页显示的 code 粘贴回终端
- 复制输出的 **refresh_token**，存入 VastAI 环境变量 `GLOBUS_REFRESH_TOKEN`

**注意**：脚本已包含 NCSA Delta 的 data_access scope；若 transfer 报 ConsentRequired，需重新运行本步骤获取新 token。

**注意**：若后续 transfer 报错需 client_secret，则改用 Confidential App，并同时保存 `GLOBUS_CLIENT_SECRET`。

```bash
Ag5zME2pvD6xm4B11bPnPmE33jvp28Yqmyr12MbXo8zMaWxN5WT4UjK4G1NzeQ7aYO8apjYJ3KyEVGyBYYOnnbvgDvGk1
```

## 步骤 3：GCP 初始化并打包凭证

在本地（Linux/macOS）执行：

```bash
# 下载 GCP
wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
tar xzf globusconnectpersonal-latest.tgz && cd globusconnectpersonal-*

# Setup（会打开浏览器，需授权一次）
./globusconnectpersonal -setup

# 启动
./globusconnectpersonal -start &
# 等待连接成功（可查看 -status）

# 打包凭证
cd /path/to/large_concept_model/vastai_onstart_scripts/globus_connection/local_prep
bash prepare_gcp_creds.sh
```

- 将生成的 `globus_creds.b64` 内容填入 `lcm_preprocess_onstart.sh` 的 `GLOBUS_CREDS` 变量（VastAI 环境变量上限 16 个，长字符串写在脚本内）

## 步骤 4：查 Endpoint UUID（可选）

- **NCSA Delta**：在 [app.globus.org/endpoints](https://app.globus.org/endpoints) 搜索 "NCSA Delta"，记下 UUID（默认已内置）
- **VastAI 侧**：`prepare_gcp_creds.sh` 会输出；也可不填，脚本会从还原的 `~/.globusonline` 自动检测

## VastAI 环境变量汇总（上限 16 个）

凭证 base64 写在 `lcm_preprocess_onstart.sh` 的 `GLOBUS_CREDS`，不占 env 配额。

| 变量                           | 必需  | 说明                                       |
| ---------------------------- | --- | ---------------------------------------- |
| `GLOBUS_REFRESH_TOKEN`       | 是   | OAuth refresh token                      |
| `GLOBUS_CLIENT_ID`           | 是   | Native App 的 client_id                   |
| `GLOBUS_CLIENT_SECRET`       | 否   | 若 Native 刷新失败，改用 Confidential App 时填写    |
| `GLOBUS_SOURCE_ENDPOINT`     | 否   | 源 endpoint UUID，不设则自动检测                  |
| `GLOBUS_DEST_ENDPOINT`       | 否   | Delta UUID                               |
| `GLOBUS_DEST_PATH`           | 否   | 目标路径                                     |
| `GLOBUS_WAIT_FOR_COMPLETION` | 否   | `true` 时等待传输完成                           |


