# 将源码展示版发布到 GitHub

1. 在电脑上将本 ZIP 解压，得到 `YOLO-GraphRAG/` 目录。
2. 在 GitHub 网站 `https://github.com/new` 建立空仓库，建议初始选择 **Private**。
   仓库名称可填写 `YOLO-GraphRAG`，不要在创建页面预先勾选 README / .gitignore / license。
3. 在 WSL2 终端进入解压后的 `YOLO-GraphRAG` 目录，确认有 `.gitignore`、`README.md`、`frontend/`、`scripts/`。
4. 依次执行：

```bash
git init -b main
git status --short
git add README.md .gitignore .env.example GITHUB_发布步骤.md
git add frontend scripts tests cypher
git add yolo_qlora_v2 yolo_qlora_eval_stage yolo_alpaca_seed YOLO_eval_v2_人工构建
git add docs requirements*.txt yolo_kg_import_key_fix.py
git diff --cached --stat
git diff --cached --name-only
git commit -m "Initial release: YOLO-GraphRAG source"
git remote add origin https://github.com/YOUR_USERNAME/YOLO-GraphRAG.git
git push -u origin main
```

5. 用你的 GitHub 用户名替换 `YOUR_USERNAME`；GitHub 推送需要完成 HTTPS/SSH 身份验证。
   推荐安装 GitHub CLI 后运行 `gh auth login`，按浏览器提示授权。不要将令牌粘贴到聊天里。
6. 如果要从 Private 改为 Public，在 GitHub 仓库 Settings 中修改前再次检查内容和许可。

本 ZIP 是源码展示版，不是完整课程提交包。原有 WSL2 项目和模型/数据库不受影响。
