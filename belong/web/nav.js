// Shared top menu for every page. Add or rename links here, once.
// Styled from the Belong. design system (NavLink, the wordmark, surface-nav-blur).
(function () {
  const LINKS = [
    { href: "/", label: "Demo", also: ["/demo"] },
    { href: "/chat", label: "In chat" },
    { href: "/trust", label: "Trust and security" },
    { href: "/inspector", label: "Inspector" },
    { href: "/trust/TRUST.md", label: "Technical details" },
  ];
  const here = location.pathname.replace(/\/+$/, "") || "/";

  const style = document.createElement("style");
  style.textContent = `
    nav.site-nav { display: block; position: sticky; top: 0; z-index: 20; margin: 0; padding: 0;
      background: var(--surface-nav-blur); -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle); }
    .site-nav .inner { max-width: var(--content-max); margin: 0 auto; padding: 0 var(--page-margin, 24px); height: 60px;
      display: flex; align-items: center; gap: var(--space-6); }
    .site-nav .wordmark { font-size: 22px; margin-right: auto; }
    .site-nav .links { display: flex; gap: var(--space-6); }
    .site-nav a.link { font: var(--text-body-s); font-weight: 500; color: var(--text-secondary); text-decoration: none;
      padding: 6px 0; border-bottom: 2px solid transparent; transition: color var(--duration-fast) var(--ease-standard); }
    .site-nav a.link:hover { color: var(--text-primary); }
    .site-nav a.link.on { color: var(--text-primary); border-bottom-color: var(--border-accent); }
    .site-nav .menu { display: none; background: transparent; border: 1px solid var(--border-strong); color: var(--text-primary);
      border-radius: var(--radius-md); padding: 6px 12px; font: var(--text-label); font-weight: 600; cursor: pointer; }
    @media (max-width: 720px) {
      .site-nav .menu { display: block; }
      .site-nav .links { display: none; position: absolute; top: 60px; left: 0; right: 0; flex-direction: column; gap: 0;
        background: var(--surface-card); border-bottom: 1px solid var(--border-default); padding: var(--space-2) var(--page-margin, 24px); }
      .site-nav .links a.link { padding: var(--space-3) 0; border-bottom: 1px solid var(--border-subtle); }
      .site-nav .links a.link.on { border-bottom-color: var(--border-subtle); color: var(--text-accent); }
      .site-nav.open .links { display: flex; }
    }
    body.only-A .site-nav, body.only-B .site-nav { display: none; }
  `;
  document.head.appendChild(style);

  const nav = document.createElement("nav");
  nav.className = "site-nav";
  nav.setAttribute("aria-label", "Site");
  nav.innerHTML = `<div class="inner">
      <a class="wordmark" href="/">Belong.</a>
      <button class="menu" aria-expanded="false">Menu</button>
      <div class="links">${LINKS.map(l => {
        const on = l.href === here || (l.also || []).includes(here);
        return `<a class="link${on ? " on" : ""}" href="${l.href}"${on ? ' aria-current="page"' : ""}>${l.label}</a>`;
      }).join("")}</div>
    </div>`;
  const button = nav.querySelector(".menu");
  button.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    button.setAttribute("aria-expanded", String(open));
  });
  document.body.prepend(nav);
})();
