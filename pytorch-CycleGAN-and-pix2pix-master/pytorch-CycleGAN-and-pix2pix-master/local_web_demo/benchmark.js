const chartDom = {
    chartStory: document.getElementById("chartStory"),
    chartSummary: document.getElementById("chartSummary"),
    chartConclusions: document.getElementById("chartConclusions"),
    datasetCharts: document.getElementById("datasetCharts"),
};
function createBullet(text) {
    const div = document.createElement("div");
    div.className = "bullet-item";
    div.textContent = text;
    return div;
}
function renderBulletList(container, items) {
    container.innerHTML = "";
    items.forEach((item) => container.appendChild(createBullet(item)));
}
function formatValue(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }
    const number = Number(value);
    if (Number.isNaN(number)) {
        return String(value);
    }
    return number.toFixed(4);
}
function maxMetric(rows, key) {
    return Math.max(...rows.map((row) => Number(row[key]) || 0), 1);
}
function championRow(rows, key) {
    return rows.reduce((best, current) => {
        if (!best) {
            return current;
        }
        return (Number(current[key]) || -Infinity) > (Number(best[key]) || -Infinity) ? current : best;
    }, null);
}
function arrowDelta(current, baseline) {
    const value = Number(current) - Number(baseline);
    if (!Number.isFinite(value)) {
        return '<span class="arrow-flat">—</span>';
    }
    if (Math.abs(value) < 1e-8) {
        return '<span class="arrow-flat">→ 持平</span>';
    }
    return value > 0
        ? `<span class="arrow-up">↑ ${value.toFixed(4)}</span>`
        : `<span class="arrow-flat">↓ ${Math.abs(value).toFixed(4)}</span>`;
}
function renderSummary(bootstrap) {
    const pills = [
        `模型预设 ${bootstrap.models.length} 个`,
        `传统方法 ${bootstrap.traditional_methods.length} 种`,
        `数据集 ${bootstrap.benchmarks.length} 组`,
        `推荐赛道 ${bootstrap.presentation_pack.track}`,
    ];
    chartDom.chartSummary.innerHTML = pills.map((item) => `<span class="metric-pill">${item}</span>`).join("");
}
function renderConclusions(bootstrap) {
    const improved = bootstrap.models.find((item) => item.key === "improved_mpcgan");
    const classic = bootstrap.models.find((item) => item.key === "classic_cyclegan");
    const euvpImproved = improved?.benchmarks?.euvp || {};
    const euvpClassic = classic?.benchmarks?.euvp || {};
    const conclusions = [
        `在 EUVP 上，改进模型的 PSNR 为 ${formatValue(euvpImproved.psnr)}，高于原始 CycleGAN 的 ${formatValue(euvpClassic.psnr)}。`,
        `在 EUVP 上，改进模型的 SSIM 为 ${formatValue(euvpImproved.ssim)}，说明结构恢复更稳定。`,
        "传统方法可以改善部分亮度或色偏，但难以同时兼顾结构保持与整体视觉自然度。",
        "图表页适合作为软件应用展示中的数据支撑页，与首页交互演示形成完整证据链。",
    ];
    renderBulletList(chartDom.chartConclusions, conclusions);
}
function renderStory() {
    renderBulletList(chartDom.chartStory, [
        "先讲 EUVP：说明主训练域上改进模型相对原始 CycleGAN 的提升。",
        "再讲 UIEB：说明跨域泛化表现，突出作品不只是局部调优，而是具备迁移价值。",
        "最后回到传统方法对照：强调你的系统不是单点模型，而是完整的软件应用平台。",
    ]);
}
function buildMetricRows(rows, metricKey, label) {
    const maxValue = maxMetric(rows, metricKey);
    const winner = championRow(rows, metricKey);
    const classic = rows.find((row) => row.model === "euvp_cyclegan_full") || rows.find((row) => row.label.includes("原始"));
    return `
        <div class="chart-metric-card">
            <h4>${label} ${winner ? `<span class="winner-badge">冠军：${winner.label}</span>` : ""}</h4>
            <div class="bar-stack">
                ${rows.map((row) => `
                    <div class="bar-row ${winner && winner.label === row.label ? "highlight-row" : ""}">
                        <div class="bar-label"><span>${row.label}</span><span>${formatValue(row[metricKey])} ${classic && row.label.includes("改进") ? arrowDelta(row[metricKey], classic[metricKey]) : ""}</span></div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:${Math.max(4, ((Number(row[metricKey]) || 0) / maxValue) * 100)}%"></div>
                        </div>
                    </div>
                `).join("")}
            </div>
        </div>
    `;
}
function renderDatasetCharts(bootstrap) {
    chartDom.datasetCharts.innerHTML = "";
    bootstrap.benchmarks.forEach((panel) => {
        const article = document.createElement("article");
        article.className = "chart-card";
        const panelWinner = championRow(panel.rows, "psnr");
        article.innerHTML = `
            <h2>${panel.title}</h2>
            <p>围绕 ${panel.dataset.toUpperCase()} 数据集，对传统方法、原始 CycleGAN 与改进模型进行多指标柱状可视化。${panelWinner ? ` 当前 PSNR 冠军为 ${panelWinner.label}。` : ""}</p>
            <div class="chart-board">
                ${buildMetricRows(panel.rows, "psnr", "PSNR")}
                ${buildMetricRows(panel.rows, "ssim", "SSIM")}
                ${buildMetricRows(panel.rows, "uciqe_pred", "UCIQE")}
                ${buildMetricRows(panel.rows, "uiqm_pred", "UIQM")}
            </div>
        `;
        chartDom.datasetCharts.appendChild(article);
    });
}
async function loadBootstrap() {
    try {
        const response = await fetch("/api/bootstrap", { cache: "no-store" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        const response = await fetch("./data/bootstrap.json", { cache: "no-store" });
        if (!response.ok) {
            throw new Error("无法读取图表展示数据。");
        }
        return await response.json();
    }
}
async function init() {
    const bootstrap = await loadBootstrap();
    renderStory();
    renderSummary(bootstrap);
    renderConclusions(bootstrap);
    renderDatasetCharts(bootstrap);
}
init().catch((error) => {
    renderBulletList(chartDom.chartConclusions, [`图表页初始化失败：${error.message}`]);
});
