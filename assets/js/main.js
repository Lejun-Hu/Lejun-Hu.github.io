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
 * 技术栈分类折叠/展开交互
 * 支持：1) 折叠宏（收起/展开整个技术栈网格）
 *       2) 单个分类折叠/展开
 *       3) 全局全部折叠/展开（控制每个分类的展开状态）
 */
document.addEventListener('DOMContentLoaded', () => {
  const grid = document.getElementById('tech-stack-grid');
  const macroToggle = document.getElementById('macro-toggle');
  const categories = document.querySelectorAll('.tech-category');
  const globalToggle = document.getElementById('toggle-all-categories');

  // 1. 折叠宏：收起/展开整个技术栈网格
  if (macroToggle && grid) {
    macroToggle.addEventListener('click', () => {
      const isExpanded = macroToggle.getAttribute('aria-expanded') === 'true';
      const newState = !isExpanded;

      macroToggle.setAttribute('aria-expanded', newState);
      const textSpan = macroToggle.querySelector('.toggle-text');
      const iconSpan = macroToggle.querySelector('.toggle-icon');
      if (textSpan) {
        textSpan.textContent = newState ? '收起一级目录' : '展开一级目录';
      }
      if (iconSpan) {
        iconSpan.textContent = newState ? '▼' : '▶';
      }

      if (newState) {
        grid.classList.remove('collapsed-macro');
      } else {
        grid.classList.add('collapsed-macro');
      }
    });
  }

  // 2. 单个分类折叠/展开
  categories.forEach(category => {
    const header = category.querySelector('.tech-category-header');
    const toggleBtn = category.querySelector('.category-toggle-btn');

    if (!header || !toggleBtn) return;

    const toggleCategory = () => {
      const isCollapsed = category.classList.contains('collapsed');
      category.classList.toggle('collapsed');
      toggleBtn.setAttribute('aria-expanded', isCollapsed);
    };

    header.addEventListener('click', toggleCategory);
  });

  // 3. 全局折叠/展开按钮：控制每个分类的展开状态
  if (globalToggle) {
    globalToggle.addEventListener('click', () => {
      const isAllExpanded = globalToggle.getAttribute('aria-expanded') === 'true';
      const newState = !isAllExpanded;

      categories.forEach(category => {
        const toggleBtn = category.querySelector('.category-toggle-btn');
        if (newState) {
          category.classList.remove('collapsed');
        } else {
          category.classList.add('collapsed');
        }
        if (toggleBtn) {
          toggleBtn.setAttribute('aria-expanded', newState);
        }
      });

      globalToggle.setAttribute('aria-expanded', newState);
      const textSpan = globalToggle.querySelector('.toggle-text');
      const iconSpan = globalToggle.querySelector('.toggle-icon');
      if (textSpan) {
        textSpan.textContent = newState ? '收起全部二级目录' : '展开全部二级目录';
      }
      if (iconSpan) {
        iconSpan.textContent = newState ? '▼' : '▶';
      }
    });
  }
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
