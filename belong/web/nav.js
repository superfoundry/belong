// Shared top menu for every page. Add or rename links here, once.
// Styled from the Belong design system (NavLink, the wordmark, surface-nav-blur).
(function () {
  const LINKS = [
    { href: "/", label: "Demo", also: ["/demo"] },
    { href: "/connector", label: "Connector", also: ["/chat"] },
    { href: "/trust", label: "Trust and security" },
    { href: "/inspector", label: "Inspector" },
    { href: "/trust/TRUST.md", label: "Technical details" },
  ];
  const here = location.pathname.replace(/\/+$/, "") || "/";

  const style = document.createElement("style");
  // Every part has a class of its own (site-nav-...) and the button and links start from all: unset, so no
  // page's styles reach them: demo.html's own .menu dropdown once dragged this button out of the bar.
  style.textContent = `
    nav.site-nav { all: unset; display: block; position: sticky; top: 0; z-index: 20;
      background: var(--surface-nav-blur); -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle); }
    .site-nav .site-nav-inner { max-width: var(--content-max); margin: 0 auto;
      padding: 0 var(--page-margin, 24px); height: 60px; display: flex; align-items: center; gap: var(--space-6); }
    .site-nav .wordmark { font-size: 22px; margin-right: auto; }
    .site-nav .site-nav-links { display: flex; gap: var(--space-6); }
    .site-nav a.site-nav-link { all: unset; cursor: pointer; font: var(--text-body-s); font-weight: 500;
      color: var(--text-secondary); padding: 6px 0; border-bottom: 2px solid transparent;
      transition: color var(--duration-fast) var(--ease-standard); }
    .site-nav a.site-nav-link:hover { color: var(--text-primary); }
    .site-nav a.site-nav-link.on { color: var(--text-primary); border-bottom-color: var(--border-accent); }
    .site-nav button.site-nav-menu { all: unset; box-sizing: border-box; display: none; flex: none; cursor: pointer;
      border: 1px solid var(--border-strong); color: var(--text-primary); border-radius: var(--radius-md);
      padding: 6px 12px; font: var(--text-label); font-weight: 600; }
    .site-nav a.site-nav-link:focus-visible, .site-nav button.site-nav-menu:focus-visible {
      outline: 2px solid var(--focus-ring); outline-offset: 2px; }
    @media (max-width: 720px) {
      .site-nav button.site-nav-menu { display: block; }
      .site-nav .site-nav-links { display: none; position: absolute; top: 60px; left: 0; right: 0; flex-direction: column;
        gap: 0; background: var(--surface-card); border-bottom: 1px solid var(--border-default);
        padding: var(--space-2) var(--page-margin, 24px); }
      .site-nav .site-nav-links a.site-nav-link { padding: var(--space-3) 0; border-bottom: 1px solid var(--border-subtle); }
      .site-nav .site-nav-links a.site-nav-link.on { border-bottom-color: var(--border-subtle); color: var(--text-accent); }
      .site-nav.open .site-nav-links { display: flex; }
    }
    body.single .site-nav { display: none; }  /* one phone on its own (demo.html ?phone=...) */
  `;
  document.head.appendChild(style);

  const nav = document.createElement("nav");
  nav.className = "site-nav";
  nav.setAttribute("aria-label", "Site");
  nav.innerHTML = `<div class="site-nav-inner">
      <a class="wordmark" href="/">Belong.</a>
      <button class="site-nav-menu" type="button" aria-expanded="false">Menu</button>
      <div class="site-nav-links">${LINKS.map(l => {
        const on = l.href === here || (l.also || []).includes(here);
        return `<a class="site-nav-link${on ? " on" : ""}" href="${l.href}"${on ? ' aria-current="page"' : ""}>${l.label}</a>`;
      }).join("")}</div>
    </div>`;
  const button = nav.querySelector(".site-nav-menu");
  button.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    button.setAttribute("aria-expanded", String(open));
  });
  document.body.prepend(nav);
})();
