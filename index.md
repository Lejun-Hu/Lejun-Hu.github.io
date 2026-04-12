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

<!-- ===== 技能部分 ===== -->
<section id="skills" class="section section-alt">
  <div class="container">
    <h2 class="section-title">专业技能</h2>

    <div class="skills-grid">
      {% for category in site.data.skills.categories %}
      <div class="skill-category">
        <h3 class="category-title">
          <span class="category-icon">{{ category.icon }}</span>
          {{ category.name }}
        </h3>

        <div class="skill-list">
          {% for skill in category.items %}
          <div class="skill-item">
            <div class="skill-info">
              <span class="skill-name">{{ skill.name }}</span>
              <span class="skill-level">{{ skill.level }}%</span>
            </div>
            <!-- 进度条 -->
            <div class="skill-bar">
              <div class="skill-progress" style="width: {{ skill.level }}%"></div>
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

<!-- ===== 项目展示部分（硬编码示例，也可放入 _data） ===== -->
<section id="projects" class="section section-alt">
  <div class="container">
    <h2 class="section-title">项目作品</h2>

    <div class="projects-grid">

      <!-- 项目 1 -->
      <article class="project-card">
        <div class="project-image">
          <img src="{{ '/assets/images/project1.jpg' | relative_url }}" 
               alt="项目截图" loading="lazy">
        </div>
        <div class="project-info">
          <h3>电商平台前端重构</h3>
          <p>使用 Next.js 重构 legacy 系统，提升性能指标，实现 SSR 渲染。</p>
          <div class="project-tags">
            <span class="tag">React</span>
            <span class="tag">Next.js</span>
            <span class="tag">TypeScript</span>
          </div>
          <a href="#" class="project-link" target="_blank">查看详情 →</a>
        </div>
      </article>

      <!-- 项目 2 -->
      <article class="project-card">
        <div class="project-image">
          <img src="{{ '/assets/images/project2.jpg' | relative_url }}" 
               alt="项目截图" loading="lazy">
        </div>
        <div class="project-info">
          <h3>数据可视化大屏</h3>
          <p>为物联网项目开发实时监控大屏，支持 WebSocket 实时数据更新。</p>
          <div class="project-tags">
            <span class="tag">Vue 3</span>
            <span class="tag">D3.js</span>
            <span class="tag">WebSocket</span>
          </div>
          <a href="#" class="project-link" target="_blank">查看详情 →</a>
        </div>
      </article>

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
