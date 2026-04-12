# 个人简历网站

这是一个基于 Jekyll 和 GitHub Pages 构建的个人简历网站模板。

## 🚀 快速开始

### 1. 使用此模板

1. 下载 `cv-website.zip` 文件并解压
2. 将所有文件上传到你的 GitHub Pages 仓库（`username.github.io`）
3. 或者 fork 此仓库并改名为 `你的用户名.github.io`

### 2. 个性化配置

编辑 `_config.yml` 文件，修改以下信息：

```yaml
title: "你的名字"
description: "你的职位 | 经验年限 | 专业领域"
email: your-email@example.com

author:
  name: 你的名字
  email: your-email@example.com
  github: your-github-username
  linkedin: your-linkedin-id
  location: 你的城市, 国家
```

### 3. 更新内容数据

#### 修改技能（`_data/skills.yml`）
```yaml
categories:
  - name: 你的技能分类
    icon: 🚀
    items:
      - name: 技能名称
        level: 90  # 百分比 0-100
```

#### 修改工作经历（`_data/experience.yml`）
```yaml
jobs:
  - company: 公司名称
    position: 职位
    location: 地点
    start_date: 2020-01
    end_date: 至今
    current: true
    description: |
      工作描述...
    highlights:
      - 成就 1
      - 成就 2
```

#### 修改教育背景（`_data/education.yml`）
```yaml
degrees:
  - school: 学校名称
    degree: 专业 学位
    start_year: 2015
    end_year: 2019
    description: 描述信息
```

### 4. 添加个人照片

将你的照片命名为 `profile.jpg` 并放入 `assets/images/` 目录。
建议尺寸：600x600 像素以上，正方形比例最佳。

### 5. 本地预览（可选）

如果你想在本地预览效果：

```bash
# 安装依赖
bundle install

# 启动本地服务器
bundle exec jekyll serve

# 访问 http://localhost:4000
```

## 📁 文件结构

```
.
├── _config.yml          # 网站配置
├── _data/               # 数据文件
│   ├── education.yml    # 教育背景
│   ├── experience.yml   # 工作经历
│   └── skills.yml       # 技能数据
├── _includes/           # 可复用组件
│   ├── footer.html      # 页脚
│   ├── head.html        # HTML head
│   └── header.html      # 导航栏
├── _layouts/            # 页面布局
│   └── default.html     # 默认布局
├── _sass/               # SCSS 样式（如需要）
├── assets/              # 静态资源
│   ├── css/
│   │   └── main.scss    # 主样式文件
│   ├── js/
│   │   └── main.js      # JavaScript 交互
│   └── images/          # 图片资源
│       ├── profile.jpg  # 你的照片（需自行添加）
│       ├── project1.jpg # 项目截图（可选）
│       └── project2.jpg # 项目截图（可选）
├── index.md             # 主页（个人简历）
└── README.md            # 本文件
```

## 🎨 自定义样式

所有样式都在 `assets/css/main.scss` 中。你可以修改 CSS 变量来改变配色：

```scss
:root {
  --color-primary: #2563eb;      // 修改主色调
  --color-text: #1e293b;         // 修改文字颜色
  --color-bg-alt: #f8fafc;       // 修改背景色
}
```

## 📝 添加博客文章（可选）

如果你想添加博客功能：

1. 创建 `_posts` 目录
2. 创建文章文件，命名格式：`YYYY-MM-DD-标题.md`
3. 添加 front matter：

```markdown
---
layout: post
title: "文章标题"
date: 2024-01-15
categories: [技术]
tags: [jekyll, 教程]
---

文章内容...
```

## 📄 生成 PDF 简历

网站已优化打印样式，你可以：
1. 在浏览器中打开网站
2. 按 `Ctrl+P`（或 `Cmd+P`）
3. 选择"另存为 PDF"
4. 隐藏页眉页脚，保存为 PDF 格式简历

## 🛠️ 技术栈

- [Jekyll](https://jekyllrb.com/) - 静态网站生成器
- [GitHub Pages](https://pages.github.com/) - 免费托管
- [SCSS](https://sass-lang.com/) - CSS 预处理器
- [Liquid](https://shopify.github.io/liquid/) - 模板语言

## 📄 许可证

MIT License - 自由使用和修改

## 💡 提示

- 修改完成后，GitHub Pages 需要 1-2 分钟来构建和部署
- 如果样式没有更新，尝试清除浏览器缓存
- 遇到问题可以查看 [GitHub Pages 文档](https://docs.github.com/cn/pages)

---

**祝求职顺利！** 🎉
