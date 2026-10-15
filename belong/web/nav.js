// Shared top menu for every page. Add or rename links here, once.
(function () {
  const LINKS = [
    { href: "/", label: "Demo" },
    { href: "/trust", label: "Trust & security" },
    { href: "/inspector", label: "Inspector" },
    { href: "/trust/TRUST.md", label: "Technical details" },
  ];
  const here = location.pathname.replace(/\/+$/, "") || "/";

  const style = document.createElement("style");
  style.textContent = `
    nav.site-nav { display: block; flex-wrap: nowrap; gap: 0; position: sticky; top: 0; z-index: 20;
      background: var(--bg, #fff); border-bottom: 1px solid var(--line, #e2dfd7); margin: 0; padding: 0; }
    nav.site-nav button.menu { border-bottom-width: 1px; }
    .site-nav .inner { max-width: 1560px; margin: 0 auto; padding: 0 16px; height: 52px;
      display: flex; align-items: center; gap: 4px; }
    .site-nav .brand { font-weight: 800; font-size: 18px; letter-spacing: -.03em; color: var(--text, #1c1b19);
      text-decoration: none; margin-right: auto; }
    .site-nav a.link { color: var(--muted, #6b6862); text-decoration: none; font-size: 14px;
      padding: 6px 10px; border-radius: 8px; white-space: nowrap; }
    .site-nav a.link:hover { color: var(--text, #1c1b19); background: var(--surface-2, rgba(0,0,0,.05)); }
    .site-nav a.link.on { color: var(--text, #1c1b19); font-weight: 600; background: var(--surface-2, rgba(0,0,0,.06)); }
    .site-nav .menu { display: none; background: none; border: 1px solid var(--line, #e2dfd7); color: var(--text, #1c1b19);
      border-radius: 8px; padding: 5px 10px; font: inherit; font-size: 14px; cursor: pointer; }
    @media (max-width: 640px) {
      .site-nav .menu { display: block; }
      .site-nav .links { display: none; position: absolute; top: 52px; left: 0; right: 0; flex-direction: column;
        background: var(--bg, #fff); border-bottom: 1px solid var(--line, #e2dfd7); padding: 8px 16px; }
      .site-nav.open .links { display: flex; }
    }
    .site-nav .links { display: flex; gap: 2px; }
    body.only-A .site-nav, body.only-B .site-nav { display: none; }
  `;
  document.head.appendChild(style);

  const nav = document.createElement("nav");
  nav.className = "site-nav";
  nav.setAttribute("aria-label", "Site");
  nav.innerHTML = `<div class="inner">
      <a class="brand" href="/">Belong</a>
      <button class="menu" aria-expanded="false">Menu</button>
      <div class="links">${LINKS.map(l => {
        const on = l.href === here;
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
