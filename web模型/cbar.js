(function () {
  function fmtPercent(v) {
    return `${Math.round(Number(v) * 100)}`;
  }

  function buildColorBar(data) {
    const root = document.getElementById("cbar");
    if (!root) return;
    if (!data || !data.cbar) return;

    const cbar = data.cbar;
    const colors = Array.isArray(cbar.colors) ? cbar.colors : [];
    const values = Array.isArray(cbar.values) ? cbar.values : [];

    root.innerHTML = "";

    const title = document.createElement("div");
    title.className = "cbar-title";
    title.textContent = cbar.title || "日照百分比";
    root.appendChild(title);

    const main = document.createElement("div");
    main.className = "cbar-main";

    const boxes = document.createElement("div");
    boxes.className = "cbar-boxes";

    const labels = document.createElement("div");
    labels.className = "cbar-labels";

    const n = Math.min(colors.length, values.length);
    for (let i = 0; i < n; i++) {
      const box = document.createElement("div");
      box.className = "cbar-box";
      box.style.background = colors[i];
      boxes.appendChild(box);

      const label = document.createElement("div");
      label.className = "cbar-label";
      label.textContent = fmtPercent(values[i]);
      labels.appendChild(label);
    }

    main.appendChild(boxes);
    main.appendChild(labels);
    root.appendChild(main);

    const unit = document.createElement("div");
    unit.className = "cbar-unit";
    unit.textContent = `单位：${cbar.unit || "%"}`;
    root.appendChild(unit);
  }

  window.buildColorBar = buildColorBar;
})();