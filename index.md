---
# Front Matter：这个页面的元数据
layout: default        # 使用 default 布局
title: 个人网页       # 页面标题
description: Egan的详细个人简历，包含工作经历、技能专长和项目展示
---

<!-- 
  注意：在 Jekyll 中，你可以混用 Markdown 和 HTML
  对于复杂布局，直接使用 HTML 更灵活
-->

<!-- ===== 个人简介部分 ===== -->
<section id="about" class="section hero-section">
  <div class="container">

    <!-- 个人照片（放在 assets/images/ 目录下） -->
    <div class="profile-image">
      <img src="{{ '/assets/images/profile.jpg' | relative_url }}" 
           alt="{{ site.author.name }}的照片" 
           loading="lazy">
    </div>

    <!-- 文字介绍 -->
    <div class="profile-content">
      <h1 class="profile-name">{{ site.author.name }}</h1>
      <p class="profile-title">{{ site.description }}</p>

      <div class="profile-summary">
        <p>
          通信工程->嵌入式工程->AI Infra ， 希望能在这个空间展示分享自己的学习心得的同时，找到潜在的合作伙伴
        </p>
      </div>

      <!-- 快速联系按钮 -->
      <div class="profile-actions">
        <a href="mailto:{{ site.author.email }}" class="btn btn-primary">
          联系我
        </a>
        <a href="#projects" class="btn btn-secondary">
          查看项目
        </a>
        <a href="{{ '/notes/' | relative_url }}" class="btn btn-secondary">
          📚 学习笔记
        </a>
      </div>
    </div>

  </div>
</section>

<!-- ===== 工作经历部分 ===== -->
<section id="experience" class="section">
  <div class="container">
    <h2 class="section-title">工作经历</h2>

    <div class="timeline">
      <!-- 
        使用 Liquid 循环遍历 _data/experience.yml 中的数据 
        site.data.experience 会自动对应 _data/experience.yml
      -->
      {% for job in site.data.experience.jobs %}
      <div class="timeline-item {% if job.current %}current{% endif %}">

        <!-- 时间标记 -->
        <div class="timeline-marker"></div>

        <!-- 内容卡片 -->
        <div class="timeline-content">
          <div class="job-header">
            <h3 class="job-company">{{ job.company }}</h3>
            <span class="job-location">{{ job.location }}</span>
          </div>

          <div class="job-meta">
            <span class="job-position">{{ job.position }}</span>
            <span class="job-date">
              {{ job.start_date }} - {{ job.end_date }}
            </span>
          </div>

          <!-- 使用 markdownify 过滤器将 Markdown 转为 HTML -->
          <div class="job-description">
            {{ job.description | markdownify }}
          </div>

          <!-- 工作亮点列表 -->
          {% if job.highlights %}
          <ul class="job-highlights">
            {% for highlight in job.highlights %}
            <li>{{ highlight }}</li>
            {% endfor %}
          </ul>
          {% endif %}
        </div>

      </div>
      {% endfor %}
    </div>

  </div>
</section>

<!-- ===== 技术栈部分 ===== -->
<section id="skills" class="section section-alt">
  <div class="container">
    <h2 class="section-title">技术栈</h2>
    <p class="tech-stack-desc">
      这里尽量列出来了我涉及到的技术栈，这样能方便我对自己的知识框架进行查漏补缺，也能更好的让合作方了解匹配情况。
    </p>

    <!-- 技术栈控制栏：折叠宏 + 全局展开 -->
    <div class="tech-stack-controls">
      <button type="button" class="macro-toggle-btn" id="macro-toggle" aria-expanded="false">
        <span class="toggle-text">展开一级目录</span>
        <span class="toggle-icon">▶</span>
      </button>
      <button type="button" class="btn btn-secondary" id="toggle-all-categories" aria-expanded="false">
        <span class="toggle-text">展开全部二级目录</span>
        <span class="toggle-icon">▶</span>
      </button>
    </div>

    <div class="tech-stack-grid collapsed-macro" id="tech-stack-grid">
      {% for category in site.data.skills.categories %}
      <div class="tech-category collapsed" data-category-index="{{ forloop.index }}">
        <div class="tech-category-header">
          <h3 class="tech-category-title">{{ category.name }}</h3>
          <button type="button" class="category-toggle-btn" aria-label="折叠分类" aria-expanded="false">
            <span class="toggle-icon">▼</span>
          </button>
        </div>

        <div class="tech-subcategories">
          {% for sub in category.subcategories %}
          <div class="tech-subcategory">
            <h4 class="tech-subcategory-title">{{ sub.name }}</h4>
            <div class="tech-skill-tags">
              {% for skill in sub.skills %}
              <span class="tech-skill-tag" data-level="{{ skill.level }}">
                <span class="skill-name">{{ skill.name }}</span>
                <span class="skill-sep">·</span>
                <span class="skill-level-text">{{ skill.level }}</span>
              </span>
              {% endfor %}
            </div>
          </div>
          {% endfor %}
        </div>

      </div>
      {% endfor %}
    </div>

  </div>
</section>

<!-- ===== 教育背景部分 ===== -->
<section id="education" class="section">
  <div class="container">
    <h2 class="section-title">教育背景</h2>

    <div class="education-list">
      {% for edu in site.data.education.degrees %}
      <div class="education-item">
        <div class="edu-years">
          {{ edu.start_year }} - {{ edu.end_year }}
        </div>
        <div class="edu-content">
          <h3>{{ edu.school }}</h3>
          <p class="edu-degree">{{ edu.degree }}</p>
          <p class="edu-desc">{{ edu.description }}</p>
        </div>
      </div>
      {% endfor %}
    </div>

  </div>
</section>

<!-- ===== 项目展示部分 ===== -->
<section id="projects" class="section section-alt">
  <div class="container">
    <h2 class="section-title">项目作品</h2>

    <div class="projects-list">
      {% for project in site.data.projects.projects %}
      <article class="project-row" data-project-index="{{ forloop.index }}">
        <!-- 标题行（始终可见） -->
        <div class="project-header">
          <div class="project-header-main">
            <h3 class="project-title">{{ project.title }}</h3>
            <span class="project-period">{{ project.period }}</span>
          </div>
          <button class="project-toggle" aria-label="展开项目详情" aria-expanded="false">
            <span class="toggle-icon">▼</span>
          </button>
        </div>

        <!-- 详情面板（默认折叠） -->
        <div class="project-details">
          <!-- 技术栈 -->
          {% if project.tech_stack %}
          <div class="project-section">
            <h4 class="project-section-title">涉及技术栈</h4>
            <div class="project-tags">
              {% for tech in project.tech_stack %}
              <span class="tag">{{ tech }}</span>
              {% endfor %}
            </div>
          </div>
          {% endif %}

          <!-- 工作内容描述 -->
          {% if project.description %}
          <div class="project-section">
            <h4 class="project-section-title">工作内容描述</h4>
            <div class="project-description">
              {{ project.description | markdownify }}
            </div>
          </div>
          {% endif %}

          <!-- 项目成果 -->
          {% if project.achievements %}
          <div class="project-section">
            <h4 class="project-section-title">项目成果</h4>
            <ul class="project-achievements">
              {% for achievement in project.achievements %}
              <li>{{ achievement }}</li>
              {% endfor %}
            </ul>
          </div>
          {% endif %}
        </div>
      </article>
      {% endfor %}
    </div>

  </div>
</section>

<!-- ===== 联系方式部分 ===== -->
<section id="contact" class="section">
  <div class="container">
    <h2 class="section-title">联系我</h2>

    <div class="contact-content">
      <p>对我的经历感兴趣？欢迎通过以下方式联系我：</p>

      <div class="contact-methods">
        <a href="mailto:{{ site.author.email }}" class="contact-item">
          <span class="contact-icon">✉️</span>
          <span>{{ site.author.email }}</span>
        </a>

        {% if site.author.github %}
        <a href="https://github.com/{{ site.author.github }}" class="contact-item" target="_blank">
          <span class="contact-icon">💻</span>
          <span>GitHub: {{ site.author.github }}</span>
        </a>
        {% endif %}
      </div>

    </div>
  </div>
</section>
