(() => {
  const indexHref = document.body.dataset.indexHref || "./";
  const pageLabel = document.body.dataset.pageLabel || "Course teaching demos";

  const makeLink = (text, className = "") => {
    const link = document.createElement("a");
    link.href = indexHref;
    link.className = className;
    link.textContent = text;
    return link;
  };

  const headerHost = document.querySelector("[data-shared-header]");
  if (headerHost) {
    const nav = document.createElement("nav");
    nav.className = "site-header";
    nav.setAttribute("aria-label", "Course navigation");
    nav.append(makeLink("← Machine Learning Demos", "site-home-link"));
    const label = document.createElement("span");
    label.className = "site-page-label";
    label.textContent = pageLabel;
    nav.append(label);
    headerHost.replaceWith(nav);
  }

  const footerHost = document.querySelector("[data-shared-footer]");
  if (footerHost) {
    footerHost.className = "site-footer";
    footerHost.append(makeLink("← All demos", "site-home-link"));
    const credit = document.createElement("span");
    credit.textContent = "01204563 Advanced Machine Learning · Assoc. Prof. Punpiti Piamsa-nga";
    footerHost.append(credit);
  }
})();
