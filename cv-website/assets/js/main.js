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
 * 技能条动画
 * 当技能区域进入视口时，触发进度条宽度动画
 */
const observerOptions = {
  threshold: 0.5,  // 当 50% 的元素可见时触发
  rootMargin: '0px'
};

const skillsObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const progressBars = entry.target.querySelectorAll('.skill-progress');
      progressBars.forEach(bar => {
        // 获取目标宽度（从 style 属性中读取）
        const width = bar.style.width;
        // 先设为 0，然后动画到目标宽度
        bar.style.width = '0';
        setTimeout(() => {
          bar.style.width = width;
        }, 100);
      });
      // 动画触发一次后停止观察
      skillsObserver.unobserve(entry.target);
    }
  });
}, observerOptions);

// 页面加载完成后启动观察
document.addEventListener('DOMContentLoaded', () => {
  const skillsSection = document.querySelector('#skills');
  if (skillsSection) {
    skillsObserver.observe(skillsSection);
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
