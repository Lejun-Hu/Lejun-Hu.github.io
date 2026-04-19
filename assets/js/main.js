/**
 * 移动端菜单切换
 * 点击汉堡按钮时显示/隐藏导航菜单
 */
function toggleMenu() {
  const nav = document.querySelector('.nav-links');
  const toggle = document.querySelector('.menu-toggle');

  nav.classList.toggle('active');
  toggle.classList.toggle('active');
}

/**
 * 滚动时头部阴影效果
 * 当页面滚动超过 50px 时，给头部添加阴影
 */
window.addEventListener('scroll', () => {
  const header = document.querySelector('.site-header');

  if (window.scrollY > 50) {
    header.style.boxShadow = '0 4px 6px -1px rgb(0 0 0 / 0.1)';
  } else {
    header.style.boxShadow = 'none';
  }
});

/**
 * 技术栈标签悬停效果
 * 为技术栈标签添加简单的交互反馈
 */
document.addEventListener('DOMContentLoaded', () => {
  const skillTags = document.querySelectorAll('.tech-skill-tag');

  skillTags.forEach(tag => {
    tag.addEventListener('mouseenter', () => {
      tag.style.transform = 'translateY(-1px)';
    });

    tag.addEventListener('mouseleave', () => {
      tag.style.transform = 'translateY(0)';
    });
  });
});

/**
 * 平滑滚动（为不支持 CSS scroll-behavior 的浏览器提供回退）
 * 当点击导航链接时，平滑滚动到对应区域
 */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });

      // 移动端点击后关闭菜单
      const nav = document.querySelector('.nav-links');
      const toggle = document.querySelector('.menu-toggle');
      nav.classList.remove('active');
      toggle.classList.remove('active');
    }
  });
});

/**
 * 项目展开/收起交互
 * 点击项目标题行时展开/收起详情面板
 */
document.addEventListener('DOMContentLoaded', () => {
  const projectRows = document.querySelectorAll('.project-row');

  projectRows.forEach(row => {
    const header = row.querySelector('.project-header');
    const toggleBtn = row.querySelector('.project-toggle');

    if (!header || !toggleBtn) return;

    const toggleProject = () => {
      const isExpanded = row.classList.contains('expanded');

      // 切换展开状态
      row.classList.toggle('expanded');
      toggleBtn.setAttribute('aria-expanded', !isExpanded);
    };

    header.addEventListener('click', toggleProject);
  });
});
