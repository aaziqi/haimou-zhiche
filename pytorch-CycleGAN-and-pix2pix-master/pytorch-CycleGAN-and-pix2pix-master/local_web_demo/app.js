const state = {
    bootstrap: null,
    enhanceFile: null,
    compareFile: null,
    currentEnhanceResult: null,
    currentDetectResult: null,
    titleRotationTimer: null,
    runtimeMode: "dynamic",
};

const dom = {
    navLinks: Array.from(document.querySelectorAll(".nav-link")),
    trackBadge: document.getElementById("trackBadge"),
    projectTitle: document.getElementById("projectTitle"),
    projectSubtitle: document.getElementById("projectSubtitle"),
    dynamicHeroKeyword: document.getElementById("dynamicHeroKeyword"),
    featuredModel: document.getElementById("featuredModel"),
    featuredTrack: document.getElementById("featuredTrack"),
    featuredSlogan: document.getElementById("featuredSlogan"),
    counterModels: document.getElementById("counterModels"),
    counterArchives: document.getElementById("counterArchives"),
    counterDatasets: document.getElementById("counterDatasets"),
    heroVisualInput: document.getElementById("heroVisualInput"),
    heroVisualOutput: document.getElementById("heroVisualOutput"),
    posterMainValue: document.getElementById("posterMainValue"),
    posterMainDesc: document.getElementById("posterMainDesc"),
    posterMetricValue: document.getElementById("posterMetricValue"),
    posterMetricDesc: document.getElementById("posterMetricDesc"),
    posterSystemValue: document.getElementById("posterSystemValue"),
    posterSystemDesc: document.getElementById("posterSystemDesc"),
    achievementHeadline: document.getElementById("achievementHeadline"),
    achievementDescription: document.getElementById("achievementDescription"),
    achievementMetricA: document.getElementById("achievementMetricA"),
    achievementMetricADesc: document.getElementById("achievementMetricADesc"),
    achievementMetricB: document.getElementById("achievementMetricB"),
    achievementMetricBDesc: document.getElementById("achievementMetricBDesc"),
    highlights: document.getElementById("highlights"),
    innovations: document.getElementById("innovations"),
    modelSelect: document.getElementById("modelSelect"),
    enhanceFile: document.getElementById("enhanceFile"),
    runEnhance: document.getElementById("runEnhance"),
    compareSlider: document.getElementById("compareSlider"),
    compareBefore: document.getElementById("compareBefore"),
    compareAfter: document.getElementById("compareAfter"),
    compareAfterMask: document.getElementById("compareAfterMask"),
    compareHandle: document.getElementById("compareHandle"),
    enhanceStatus: document.getElementById("enhanceStatus"),
    enhanceMeta: document.getElementById("enhanceMeta"),
    enhanceMetrics: document.getElementById("enhanceMetrics"),
    enhanceReportLink: document.getElementById("enhanceReportLink"),
    enhanceCsvLink: document.getElementById("enhanceCsvLink"),
    downloadCurrentResult: document.getElementById("downloadCurrentResult"),
    exportPresentationShot: document.getElementById("exportPresentationShot"),
    detectorSelect: document.getElementById("detectorSelect"),
    detectConf: document.getElementById("detectConf"),
    detectConfValue: document.getElementById("detectConfValue"),
    runDetect: document.getElementById("runDetect"),
    detectStatus: document.getElementById("detectStatus"),
    detectVis: document.getElementById("detectVis"),
    detectSummary: document.getElementById("detectSummary"),
    detectList: document.getElementById("detectList"),
    detectReportLink: document.getElementById("detectReportLink"),
    detectRawJsonLink: document.getElementById("detectRawJsonLink"),
    detectEnhJsonLink: document.getElementById("detectEnhJsonLink"),
    downloadDetectVis: document.getElementById("downloadDetectVis"),
    traditionalSelect: document.getElementById("traditionalSelect"),
    compareModelSelect: document.getElementById("compareModelSelect"),
    compareFile: document.getElementById("compareFile"),
    runCompare: document.getElementById("runCompare"),
    comparisonCards: document.getElementById("comparisonCards"),
    compareStatus: document.getElementById("compareStatus"),
    compareReportLink: document.getElementById("compareReportLink"),
    benchmarkHighlights: document.getElementById("benchmarkHighlights"),
    benchmarkPanels: document.getElementById("benchmarkPanels"),
    archivePreviewGrid: document.getElementById("archivePreviewGrid"),
    presentationMeta: document.getElementById("presentationMeta"),
    pptOutline: document.getElementById("pptOutline"),
    demoScript: document.getElementById("demoScript"),
    deployModeBanner: document.getElementById("deployModeBanner"),
};

function setStatus(target, text, kind = "") {
    target.textContent = text;
    target.className = `status-chip${kind ? ` ${kind}` : ""}`;
}
function isStaticMode() {
    return state.runtimeMode === "static";
}
function showDeployModeBanner(message) {
    if (!dom.deployModeBanner) {
        return;
    }
    dom.deployModeBanner.hidden = false;
    dom.deployModeBanner.textContent = message;
}

function renderBulletList(container, items) {
    if (!container) {
        return;
    }
    container.innerHTML = "";
    items.forEach((item) => {
        const div = document.createElement("div");
        div.className = "bullet-item";
        div.textContent = item;
        container.appendChild(div);
    });
}

function renderOptions(select, options, labelKey = "label", valueKey = "key") {
    select.innerHTML = "";
    options.forEach((item) => {
        const option = document.createElement("option");
        option.value = item[valueKey];
        option.textContent = item[labelKey];
        select.appendChild(option);
    });
}

function setActiveNavFromHash() {
    const current = window.location.hash || "#overview";
    dom.navLinks.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === current);
    });
}

function updateCompareSlider(value) {
    dom.compareAfterMask.style.width = `${value}%`;
    dom.compareHandle.style.left = `${value}%`;
}

function formatMetric(value) {
    if (value === null || value === undefined || value === "") {
        return "—";
    }
    const num = Number(value);
    if (Number.isNaN(num)) {
        return String(value);
    }
    return num > 0 ? `+${num.toFixed(4)}` : num.toFixed(4);
}

function setLink(link, href) {
    if (href) {
        link.href = href;
        link.classList.remove("disabled");
    } else {
        link.href = "#";
        link.classList.add("disabled");
    }
}

function triggerDownload(url, filename) {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
}

function clampPercent(value) {
    return Math.max(0, Math.min(100, Number(value) || 0));
}

function resetDetectPanel(message = "等待检测结果") {
    state.currentDetectResult = null;
    dom.detectVis.removeAttribute("src");
    dom.detectSummary.className = "meta-panel empty";
    dom.detectSummary.textContent = message;
    dom.detectList.innerHTML = "";
    setLink(dom.detectReportLink, "");
    setLink(dom.detectRawJsonLink, "");
    setLink(dom.detectEnhJsonLink, "");
    dom.downloadDetectVis.disabled = true;
    dom.runDetect.disabled = !(state.bootstrap?.detectors?.length);
    setStatus(dom.detectStatus, dom.runDetect.disabled ? "检测器未就绪" : "等待检测", dom.runDetect.disabled ? "error" : "");
}

function animateNumber(element, target, suffix = "", duration = 1200) {
    const start = performance.now();
    const initial = Number((element.dataset.value || "0").replace(/[^\d.-]/g, "")) || 0;
    const tick = (now) => {
        const progress = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = initial + (target - initial) * eased;
        element.textContent = `${Math.round(value)}${suffix}`;
        if (progress < 1) {
            requestAnimationFrame(tick);
        } else {
            element.dataset.value = String(target);
            element.textContent = `${target}${suffix}`;
        }
    };
    requestAnimationFrame(tick);
}

function startTitleRotation(items) {
    const values = items.filter(Boolean);
    if (!values.length || !dom.dynamicHeroKeyword) {
        return;
    }
    let index = 0;
    dom.dynamicHeroKeyword.textContent = values[0];
    if (state.titleRotationTimer) {
        clearInterval(state.titleRotationTimer);
    }
    state.titleRotationTimer = setInterval(() => {
        index = (index + 1) % values.length;
        dom.dynamicHeroKeyword.animate([
            { opacity: 0.15, transform: "translateY(10px)" },
            { opacity: 1, transform: "translateY(0)" },
        ], { duration: 420, easing: "ease-out" });
        dom.dynamicHeroKeyword.textContent = values[index];
    }, 2600);
}

function registerRevealAnimations() {
    const targets = Array.from(document.querySelectorAll(".card, .achievement-card, .chart-preview-card, .archive-card"));
    targets.forEach((target, index) => {
        target.classList.add("reveal");
        target.style.transitionDelay = `${Math.min(index * 35, 240)}ms`;
    });
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add("in-view");
            }
        });
    }, { threshold: 0.18 });
    targets.forEach((target) => observer.observe(target));
}

function registerParallaxCards() {
    const panels = document.querySelectorAll(".hero, .poster-hero");
    panels.forEach((panel) => {
        panel.classList.add("parallax-card");
        panel.addEventListener("mousemove", (event) => {
            const rect = panel.getBoundingClientRect();
            const x = (event.clientX - rect.left) / rect.width;
            const y = (event.clientY - rect.top) / rect.height;
            panel.style.setProperty("--tilt-y", `${(x - 0.5) * 5.2}deg`);
            panel.style.setProperty("--tilt-x", `${(0.5 - y) * 5.2}deg`);
            panel.style.setProperty("--mx", `${(x * 100).toFixed(2)}%`);
            panel.style.setProperty("--my", `${(y * 100).toFixed(2)}%`);
        });
        panel.addEventListener("mouseleave", () => {
            panel.style.setProperty("--tilt-y", "0deg");
            panel.style.setProperty("--tilt-x", "0deg");
            panel.style.setProperty("--mx", "50%");
            panel.style.setProperty("--my", "50%");
        });
    });
}

function enhancePremiumStyling() {
    document.querySelectorAll(".achievement-card, .poster-panel, .chart-preview-card").forEach((item, index) => {
        item.classList.add("float-gentle");
        item.style.animationDelay = `${index * 0.35}s`;
    });
    document.querySelectorAll(".primary-link, .primary-button").forEach((button) => {
        button.classList.add("shine");
    });
}

async function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function apiRequest(path, body) {
    if (isStaticMode()) {
        throw new Error("当前为线上展示版，请在本地完整版中使用实时推理功能。");
    }
    const response = await fetch(path, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || "请求失败");
    }
    return payload;
}

function renderEnhanceResult(result) {
    state.currentEnhanceResult = result;
    dom.compareBefore.src = result.input_url;
    dom.compareAfter.src = result.output_url;
    updateCompareSlider(dom.compareSlider.value);
    dom.enhanceMeta.className = "meta-panel";
    dom.enhanceMeta.innerHTML = `
        <span class="meta-title">${result.model.label}</span>
        <div class="meta-line">模型说明：${result.model.description}</div>
        <div class="meta-line">Checkpoint：${result.model.checkpoint_name} @ ${result.model.epoch}</div>
        <div class="meta-line">处理耗时：${result.processing_seconds} 秒</div>
        <div class="meta-line">案例标识：${result.key}</div>
    `;
    dom.enhanceMetrics.innerHTML = `
        <div class="mini-metric"><span>输出 UCIQE</span><strong>${formatMetric(result.output_uciqe)}</strong></div>
        <div class="mini-metric"><span>输出 UIQM</span><strong>${formatMetric(result.output_uiqm)}</strong></div>
        <div class="mini-metric"><span>UCIQE 提升</span><strong>${formatMetric(result.delta_uciqe)}</strong></div>
        <div class="mini-metric"><span>UIQM 提升</span><strong>${formatMetric(result.delta_uiqm)}</strong></div>
    `;
    setLink(dom.enhanceReportLink, result.report_url);
    setLink(dom.enhanceCsvLink, result.metrics_url);
    dom.downloadCurrentResult.disabled = false;
    dom.exportPresentationShot.disabled = false;
}

function summarizeByClass(dets) {
    const buckets = new Map();
    (dets || []).forEach((d) => {
        const name = d.cls_name || String(d.cls_id ?? "cls");
        const prev = buckets.get(name) || { count: 0, maxConf: 0 };
        prev.count += 1;
        prev.maxConf = Math.max(prev.maxConf, Number(d.conf) || 0);
        buckets.set(name, prev);
    });
    return Array.from(buckets.entries())
        .map(([name, v]) => ({ name, ...v }))
        .sort((a, b) => b.count - a.count || b.maxConf - a.maxConf)
        .slice(0, 8);
}

function renderDetectCompareResult(payload) {
    state.currentDetectResult = payload;
    dom.detectVis.src = payload.compare_vis_url;
    const rawDets = payload.raw_detections || [];
    const enhDets = payload.enhanced_detections || [];
    const detector = payload.detector || {};
    const delta = enhDets.length - rawDets.length;
    dom.detectSummary.className = "meta-panel";
    dom.detectSummary.innerHTML = `
        <span class="meta-title">${detector.label || "下游检测器"}</span>
        <div class="meta-line">原图目标数：${rawDets.length}，增强图目标数：${enhDets.length}（${delta >= 0 ? "+" : ""}${delta}）</div>
        <div class="meta-line">置信度阈值：${Number(payload.conf ?? dom.detectConf.value).toFixed(2)}</div>
    `;
    dom.detectList.innerHTML = "";
    const rawTop = summarizeByClass(rawDets);
    const enhTop = summarizeByClass(enhDets);
    if (!rawTop.length && !enhTop.length) {
        const empty = document.createElement("div");
        empty.className = "bullet-item";
        empty.textContent = "原图与增强图均未检测到目标，可尝试降低置信度阈值后重新运行。";
        dom.detectList.appendChild(empty);
    } else {
        const makeSection = (title, items) => {
            const header = document.createElement("div");
            header.className = "bullet-item";
            header.textContent = title;
            dom.detectList.appendChild(header);
            if (!items.length) {
                const row = document.createElement("div");
                row.className = "detect-item";
                row.innerHTML = `<div>无目标</div><span>—</span>`;
                dom.detectList.appendChild(row);
                return;
            }
            items.forEach((it) => {
                const row = document.createElement("div");
                row.className = "detect-item";
                row.innerHTML = `<div>${it.name} × ${it.count}</div><span>${it.maxConf.toFixed(2)}</span>`;
                dom.detectList.appendChild(row);
            });
        };
        makeSection("原图 Top 类别（数量 / 最高置信度）", rawTop);
        makeSection("增强图 Top 类别（数量 / 最高置信度）", enhTop);
    }
    setLink(dom.detectReportLink, payload.report_url);
    setLink(dom.detectRawJsonLink, payload.raw_json_url);
    setLink(dom.detectEnhJsonLink, payload.enhanced_json_url);
    dom.downloadDetectVis.disabled = false;
}

function renderComparisonCards(result) {
    dom.comparisonCards.innerHTML = "";
    const inputCard = document.createElement("article");
    inputCard.className = "comparison-card";
    inputCard.innerHTML = `
        <div class="comparison-badge">输入图像</div>
        <h3>原图像</h3>
        <p>作为传统方法、原始 CycleGAN 与改进模型的统一输入。</p>
        <img src="${result.input_url}" alt="输入图像">
    `;
    dom.comparisonCards.appendChild(inputCard);

    result.variants.forEach((variant) => {
        const metrics = variant.metrics || {};
        const card = document.createElement("article");
        card.className = "comparison-card";
        const reportAction = variant.report_url
            ? `<div class="link-row"><a class="ghost-link" href="${variant.report_url}" target="_blank" rel="noopener">打开单模型报告</a></div>`
            : "";
        card.innerHTML = `
            <div class="comparison-badge">${variant.badge}</div>
            <h3>${variant.label}</h3>
            <p>${variant.description}</p>
            <img src="${variant.image_url}" alt="${variant.label}">
            <div class="comparison-metrics">
                <div class="comparison-metric"><span>输出 UCIQE</span><strong>${formatMetric(metrics.output_uciqe)}</strong></div>
                <div class="comparison-metric"><span>输出 UIQM</span><strong>${formatMetric(metrics.output_uiqm)}</strong></div>
                <div class="comparison-metric"><span>UCIQE 提升</span><strong>${formatMetric(metrics.delta_uciqe)}</strong></div>
                <div class="comparison-metric"><span>UIQM 提升</span><strong>${formatMetric(metrics.delta_uiqm)}</strong></div>
            </div>
            ${reportAction}
        `;
        dom.comparisonCards.appendChild(card);
    });
}

function renderBenchmarks(panels) {
    dom.benchmarkPanels.innerHTML = "";
    panels.forEach((panel) => {
        const article = document.createElement("article");
        article.className = "benchmark-panel";
        const rows = panel.rows.map((row) => `
            <tr>
                <td>${row.label}</td>
                <td>${row.psnr ?? "—"}</td>
                <td>${row.ssim ?? "—"}</td>
                <td>${row.uciqe_pred ?? "—"}</td>
                <td>${row.uiqm_pred ?? "—"}</td>
            </tr>
        `).join("");
        article.innerHTML = `
            <h3>${panel.title}</h3>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>方法</th>
                        <th>PSNR</th>
                        <th>SSIM</th>
                        <th>UCIQE</th>
                        <th>UIQM</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
        dom.benchmarkPanels.appendChild(article);
    });
}

function renderBenchmarkHighlights(panels) {
    dom.benchmarkHighlights.innerHTML = "";
    panels.forEach((panel) => {
        const psnrMax = Math.max(...panel.rows.map((row) => Number(row.psnr) || 0), 1);
        const ssimMax = Math.max(...panel.rows.map((row) => Number(row.ssim) || 0), 1);
        const article = document.createElement("article");
        article.className = "chart-preview-card";
        article.innerHTML = `
            <h3>${panel.title}</h3>
            <p>用图表方式快速展示代表方法在 ${panel.dataset.toUpperCase()} 数据集上的表现差异。</p>
            <div class="bar-stack">
                ${panel.rows.map((row) => `
                    <div class="bar-row">
                        <div class="bar-label"><span>${row.label} · PSNR</span><span>${row.psnr ?? "—"}</span></div>
                        <div class="bar-track"><div class="bar-fill" style="width:${clampPercent((Number(row.psnr) || 0) / psnrMax * 100)}%"></div></div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label"><span>${row.label} · SSIM</span><span>${row.ssim ?? "—"}</span></div>
                        <div class="bar-track"><div class="bar-fill" style="width:${clampPercent((Number(row.ssim) || 0) / ssimMax * 100)}%"></div></div>
                    </div>
                `).join("")}
            </div>
        `;
        dom.benchmarkHighlights.appendChild(article);
    });
    const summaryCard = document.createElement("article");
    summaryCard.className = "chart-preview-card";
    summaryCard.innerHTML = `
        <h3>深入分析</h3>
        <p>打开完整图表页可查看更细的柱状对比、指标层次和数据集迁移表现。</p>
        <div class="link-row" style="margin-top:16px;">
            <a class="ghost-link" href="./benchmark.html" target="_blank" rel="noopener">查看完整图表页</a>
        </div>
    `;
    dom.benchmarkHighlights.appendChild(summaryCard);
}

function renderArchiveCards(container, archives, emptyText = "暂无归档案例，先运行一次增强或综合对比即可自动生成。") {
    container.innerHTML = "";
    if (!archives || archives.length === 0) {
        const empty = document.createElement("div");
        empty.className = "bullet-item";
        empty.textContent = emptyText;
        container.appendChild(empty);
        return;
    }
    archives.slice(0, 6).forEach((entry) => {
        const article = document.createElement("article");
        article.className = "archive-card";
        const tags = (entry.tags || []).map((tag) => `<span class="micro-badge">${tag}</span>`).join("");
        article.innerHTML = `
            <img src="${entry.cover_url}" alt="${entry.title}">
            <div class="archive-meta"><span>${entry.created_at}</span><span>${entry.model_label || "案例"}</span></div>
            <h3>${entry.title}</h3>
            <p>${entry.summary}</p>
            <div class="badge-row">${tags}</div>
            <div class="link-row">
                <a class="ghost-link" href="${entry.report_url}" target="_blank" rel="noopener">打开案例</a>
            </div>
        `;
        container.appendChild(article);
    });
}

function renderAchievementCards() {
    const featured = state.bootstrap.models.find((item) => item.key === "improved_mpcgan") || state.bootstrap.models[0];
    const euvp = featured?.benchmarks?.euvp || {};
    const uieb = featured?.benchmarks?.uieb || {};
    const latestArchive = state.bootstrap.archives?.[0];
    dom.achievementHeadline.textContent = featured ? featured.label : "主推模型";
    dom.achievementDescription.textContent = featured ? `${featured.description.replace(/[。！!？?]+$/, "")}，当前适合作为首页主展示模型。` : "正在读取主推模型信息。";
    dom.achievementMetricA.textContent = `EUVP PSNR ${euvp.psnr ?? "?"} / SSIM ${euvp.ssim ?? "?"}`;
    dom.achievementMetricADesc.textContent = `UIEB 泛化表现：PSNR ${uieb.psnr ?? "?"} / SSIM ${uieb.ssim ?? "?"}。`;
    dom.achievementMetricB.textContent = `${state.bootstrap.models.length} 个模型 + ${state.bootstrap.traditional_methods.length} 种传统方法`;
    dom.achievementMetricBDesc.textContent = "覆盖上传增强、方法对比、自动报告、图表页与结果归档。";
    dom.posterMainValue.textContent = featured ? featured.label : "主推模型";
    dom.posterMainDesc.textContent = featured ? `${featured.description.replace(/[。！!？?]+$/, "")}，适合作为首页封面区的核心展示成果。` : "正在读取主推模型信息。";
    dom.posterMetricValue.textContent = `PSNR ${euvp.psnr ?? "?"} / SSIM ${euvp.ssim ?? "?"}`;
    dom.posterMetricDesc.textContent = "主训练域 EUVP 表现突出，可与 UIEB 泛化结果联动讲解。";
    dom.posterSystemValue.textContent = "上传增强 + 对比展示 + 图表分析";
    dom.posterSystemDesc.textContent = `已自动归档 ${state.bootstrap.archives?.length ?? 0} 个案例，形成可复用的作品案例库。`;
    if (latestArchive) {
        dom.heroVisualInput.src = latestArchive.input_url || latestArchive.cover_url;
        dom.heroVisualOutput.src = latestArchive.cover_url;
    } else {
        dom.heroVisualInput.removeAttribute("src");
        dom.heroVisualOutput.removeAttribute("src");
    }
    animateNumber(dom.counterModels, state.bootstrap.models.length);
    animateNumber(dom.counterArchives, state.bootstrap.archives?.length || 0);
    animateNumber(dom.counterDatasets, state.bootstrap.benchmarks.length);
    startTitleRotation([
        "智能图像增强平台",
        "软件应用展示系统",
        "水下视觉可用性提升引擎",
        "交互式软件应用展示前端",
    ]);
}

function renderPresentationPack(pack) {
    dom.trackBadge.textContent = pack.track;
    dom.projectTitle.textContent = pack.project_name;
    dom.projectSubtitle.textContent = pack.project_subtitle;
    dom.featuredTrack.textContent = pack.track;
    dom.featuredSlogan.textContent = pack.slogan;
    const featured = state.bootstrap.models.find((item) => item.key === "improved_mpcgan") || state.bootstrap.models[0];
    dom.featuredModel.textContent = featured ? featured.label : "可用模型";
    renderBulletList(dom.highlights, pack.highlights);
    renderBulletList(dom.innovations, pack.innovations);
    renderBulletList(dom.presentationMeta, [
        `作品名称：${pack.project_name}`,
        `副标题：${pack.project_subtitle}`,
        `参加赛道：${pack.track}`,
        `传播口号：${pack.slogan}`,
    ]);
    renderBulletList(dom.pptOutline, pack.ppt_outline);
    renderBulletList(dom.demoScript, pack.demo_script);
    renderAchievementCards();
}


function disableInteractiveRuntime() {
    showDeployModeBanner("当前为线上展示版：可查看作品介绍、图表、案例与报告；实时增强、综合对比和检测功能仅在本地完整版开放。");
    if (dom.runEnhance) {
        dom.runEnhance.disabled = true;
        dom.runEnhance.textContent = "线上展示版不开放实时增强";
    }
    if (dom.runCompare) {
        dom.runCompare.disabled = true;
        dom.runCompare.textContent = "线上展示版不开放实时对比";
    }
    if (dom.runDetect) {
        dom.runDetect.disabled = true;
    }
    if (dom.enhanceFile) {
        dom.enhanceFile.disabled = true;
    }
    if (dom.compareFile) {
        dom.compareFile.disabled = true;
    }
    setStatus(dom.enhanceStatus, "线上展示版：请查看预生成案例与报告", "loading");
    setStatus(dom.compareStatus, "线上展示版：请查看预生成综合对比案例", "loading");
    setStatus(dom.detectStatus, "线上展示版：检测对照仅在本地完整版开放", "loading");
    resetDetectPanel("线上展示版不提供实时检测，请查看归档中的对照案例。");
}
function hydrateStaticPreview() {
    const enhanceEntry = state.bootstrap?.archives?.find((item) => item.type === "enhance");
    const compareEntry = state.bootstrap?.archives?.find((item) => item.type === "compare");
    const detectEntry = state.bootstrap?.archives?.find((item) => item.type === "detect_compare");
    if (enhanceEntry) {
        renderEnhanceResult({
            input_url: enhanceEntry.input_url,
            output_url: enhanceEntry.cover_url,
            report_url: enhanceEntry.report_url,
            metrics_url: enhanceEntry.metrics_url,
            processing_seconds: "预生成案例",
            key: enhanceEntry.id,
            output_uciqe: null,
            output_uiqm: null,
            delta_uciqe: null,
            delta_uiqm: null,
            model: {
                label: enhanceEntry.model_label || "改进模型 MPCGAN",
                description: "线上展示版使用预生成增强案例，实时推理功能保留在本地完整版。",
                checkpoint_name: "euvp_mpcgan_stage2_s0",
                epoch: "202",
            },
        });
    }
    if (compareEntry) {
        const classic = state.bootstrap?.models?.find((item) => item.key === "classic_cyclegan");
        const improved = state.bootstrap?.models?.find((item) => item.key === "improved_mpcgan");
        renderComparisonCards({
            input_url: compareEntry.input_url,
            variants: [
                {
                    badge: "传统方法",
                    label: "传统方法 Gray World + CLAHE",
                    description: "作为基础增强方案，适合展示颜色校正与局部对比度提升效果。",
                    image_url: compareEntry.input_url,
                    metrics: {},
                    report_url: "",
                },
                {
                    badge: classic?.badge || "基线模型",
                    label: classic?.label || "原 CycleGAN",
                    description: classic?.description || "无配对图像翻译基线模型。",
                    image_url: compareEntry.cover_url,
                    metrics: {},
                    report_url: "",
                },
                {
                    badge: improved?.badge || "主推模型",
                    label: improved?.label || compareEntry.model_label || "改进模型 MPCGAN",
                    description: improved?.description || "当前适合作为主推方案的增强模型。",
                    image_url: compareEntry.cover_url,
                    metrics: {},
                    report_url: compareEntry.report_url,
                },
            ],
        });
        setLink(dom.compareReportLink, compareEntry.report_url);
    }
    if (detectEntry) {
        renderDetectCompareResult({
            compare_vis_url: detectEntry.cover_url,
            raw_detections: [],
            enhanced_detections: [],
            detector: {
                label: detectEntry.model_label || "下游检测器 (YOLO)",
            },
            report_url: detectEntry.report_url,
            raw_json_url: detectEntry.extra?.raw_json_url || "",
            enhanced_json_url: detectEntry.extra?.enhanced_json_url || detectEntry.metrics_url || "",
            conf: Number(dom.detectConf?.value ?? 0.25),
        });
    }
}
async function bootstrap() {
    let payload;
    try {
        const response = await fetch("/api/bootstrap", { cache: "no-store" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        payload = await response.json();
        state.runtimeMode = payload.runtime_mode || "dynamic";
    } catch (error) {
        const response = await fetch("./data/bootstrap.json", { cache: "no-store" });
        if (!response.ok) {
            throw new Error("无法读取展示数据，请检查 data/bootstrap.json 是否存在。");
        }
        payload = await response.json();
        state.runtimeMode = "static";
    }
    state.bootstrap = payload;
    renderOptions(dom.modelSelect, payload.models);
    renderOptions(dom.compareModelSelect, payload.models.filter((item) => item.key !== "classic_cyclegan"));
    renderOptions(dom.traditionalSelect, payload.traditional_methods);
    if (dom.detectorSelect && Array.isArray(payload.detectors) && payload.detectors.length) {
        renderOptions(dom.detectorSelect, payload.detectors);
        dom.detectorSelect.value = payload.detectors[0].key;
        resetDetectPanel();
    } else {
        if (dom.detectorSelect) {
            dom.detectorSelect.innerHTML = "";
        }
        resetDetectPanel("检测器未就绪，请先准备 YOLO 权重并重启服务。");
    }
    if (payload.models.some((item) => item.key === "improved_mpcgan")) {
        dom.modelSelect.value = "improved_mpcgan";
    }
    if (payload.traditional_methods.some((item) => item.key === "grayworld_clahe")) {
        dom.traditionalSelect.value = "grayworld_clahe";
    }
    if ([...dom.compareModelSelect.options].some((option) => option.value === "improved_mpcgan")) {
        dom.compareModelSelect.value = "improved_mpcgan";
    }
    renderBenchmarkHighlights(payload.benchmarks);
    renderBenchmarks(payload.benchmarks);
    renderPresentationPack(payload.presentation_pack);
    renderArchiveCards(dom.archivePreviewGrid, payload.archives || []);
    if (isStaticMode()) {
        disableInteractiveRuntime();
        hydrateStaticPreview();
    }
}

async function loadImage(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error(`图片加载失败：${url}`));
        image.src = url;
    });
}

async function exportPresentationShot() {
    if (!state.currentEnhanceResult) {
        setStatus(dom.enhanceStatus, "请先完成一次增强演示。", "error");
        return;
    }
    try {
        setStatus(dom.enhanceStatus, "正在导出演示截图", "loading");
        const result = state.currentEnhanceResult;
        const [beforeImage, afterImage] = await Promise.all([
            loadImage(result.input_url),
            loadImage(result.output_url),
        ]);
        const canvas = document.createElement("canvas");
        canvas.width = 1600;
        canvas.height = 900;
        const ctx = canvas.getContext("2d");
        const gradient = ctx.createLinearGradient(0, 0, 1600, 900);
        gradient.addColorStop(0, "#07111f");
        gradient.addColorStop(1, "#123b67");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#ffffff";
        ctx.font = "700 46px Microsoft YaHei";
        ctx.fillText("海眸智澈 · 水下图像增强演示截图", 70, 88);
        ctx.fillStyle = "#b9d2ef";
        ctx.font = "26px Microsoft YaHei";
        ctx.fillText(`模型：${result.model.label}    Checkpoint：${result.model.checkpoint_name} @ ${result.model.epoch}`, 70, 130);

        const panelWidth = 660;
        const panelHeight = 460;
        const topY = 180;
        const leftX = 70;
        const rightX = 870;
        [leftX, rightX].forEach((x) => {
            ctx.fillStyle = "rgba(255,255,255,0.06)";
            ctx.fillRect(x, topY, panelWidth, panelHeight);
        });
        const drawFit = (image, x, y, w, h) => {
            const scale = Math.min(w / image.width, h / image.height);
            const drawW = image.width * scale;
            const drawH = image.height * scale;
            const drawX = x + (w - drawW) / 2;
            const drawY = y + (h - drawH) / 2;
            ctx.drawImage(image, drawX, drawY, drawW, drawH);
        };
        ctx.fillStyle = "#dff5ff";
        ctx.font = "700 24px Microsoft YaHei";
        ctx.fillText("原图像", leftX, 170);
        ctx.fillText("增强结果", rightX, 170);
        drawFit(beforeImage, leftX + 12, topY + 12, panelWidth - 24, panelHeight - 24);
        drawFit(afterImage, rightX + 12, topY + 12, panelWidth - 24, panelHeight - 24);

        const metricCards = [
            ["输出 UCIQE", formatMetric(result.output_uciqe)],
            ["输出 UIQM", formatMetric(result.output_uiqm)],
            ["UCIQE 提升", formatMetric(result.delta_uciqe)],
            ["UIQM 提升", formatMetric(result.delta_uiqm)],
        ];
        metricCards.forEach(([label, value], index) => {
            const x = 70 + (index % 2) * 400;
            const y = 700 + Math.floor(index / 2) * 90;
            ctx.fillStyle = "rgba(255,255,255,0.07)";
            ctx.fillRect(x, y, 340, 72);
            ctx.fillStyle = "#9fc0e3";
            ctx.font = "20px Microsoft YaHei";
            ctx.fillText(label, x + 18, y + 28);
            ctx.fillStyle = "#ffffff";
            ctx.font = "700 28px Microsoft YaHei";
            ctx.fillText(value, x + 18, y + 58);
        });
        ctx.fillStyle = "#a8bfdc";
        ctx.font = "24px Microsoft YaHei";
        ctx.fillText("适用场景：展示增强前后差异，适合导出到 PPT、海报或汇报材料。", 870, 720);
        ctx.fillText(`报告链接：${result.report_url}`, 870, 768);
        const dataUrl = canvas.toDataURL("image/png");
        triggerDownload(dataUrl, `presentation_shot_${Date.now()}.png`);
        setStatus(dom.enhanceStatus, "演示截图已导出", "success");
    } catch (error) {
        setStatus(dom.enhanceStatus, error.message, "error");
    }
}

function downloadCurrentResult() {
    if (!state.currentEnhanceResult) {
        setStatus(dom.enhanceStatus, "请先完成一次增强演示。", "error");
        return;
    }
    const suffix = state.currentEnhanceResult.output_url.split(".").pop()?.split("?")[0] || "png";
    triggerDownload(state.currentEnhanceResult.output_url, `enhanced_result.${suffix}`);
}

async function runEnhance() {
    if (!state.enhanceFile) {
        setStatus(dom.enhanceStatus, "请先选择图片", "error");
        return;
    }
    try {
        dom.runEnhance.disabled = true;
        setStatus(dom.enhanceStatus, "增强处理中...", "loading");
        resetDetectPanel();
        const image = await fileToDataUrl(state.enhanceFile);
        const payload = await apiRequest("/api/enhance", {
            filename: state.enhanceFile.name,
            image,
            model_key: dom.modelSelect.value,
        });
        renderEnhanceResult(payload.result);
        dom.runDetect.disabled = !(state.bootstrap?.detectors?.length);
        if (!dom.runDetect.disabled) {
            await runDetect(true);
        }
        const archiveResponse = await fetch("/api/archive");
        const archivePayload = await archiveResponse.json();
        state.bootstrap.archives = archivePayload.archives || [];
        renderArchiveCards(dom.archivePreviewGrid, archivePayload.archives || []);
        renderAchievementCards();
        setStatus(dom.enhanceStatus, "增强完成", "success");
    } catch (error) {
        setStatus(dom.enhanceStatus, error.message, "error");
    } finally {
        dom.runEnhance.disabled = false;
    }
}

async function runDetect(silent = false) {
    const fallback = state.bootstrap?.archives?.find((item) => item?.type === "enhance" && item?.input_url && item?.cover_url) || null;
    const rawUrl = state.currentEnhanceResult?.input_url || fallback?.input_url || "";
    const enhUrl = state.currentEnhanceResult?.output_url || fallback?.cover_url || "";
    if (!rawUrl || !enhUrl) {
        if (!silent) {
            setStatus(dom.detectStatus, "请先完成一次增强演示。", "error");
        }
        return;
    }
    if (!state.bootstrap?.detectors?.length) {
        if (!silent) {
            setStatus(dom.detectStatus, "检测器未就绪。", "error");
        }
        return;
    }
    try {
        dom.runDetect.disabled = true;
        setStatus(dom.detectStatus, "检测对照中...", "loading");
        const conf = Number(dom.detectConf?.value ?? 0.25);
        const payload = await apiRequest("/api/detect_compare", {
            raw_image_url: rawUrl,
            enhanced_image_url: enhUrl,
            detector_key: dom.detectorSelect.value,
            conf,
            imgsz: 640,
        });
        renderDetectCompareResult({ ...payload, conf });
        setStatus(dom.detectStatus, "检测对照完成", "success");
    } catch (error) {
        setStatus(dom.detectStatus, error.message, "error");
    } finally {
        dom.runDetect.disabled = false;
    }
}

function downloadDetectVis() {
    if (!state.currentDetectResult?.compare_vis_url) {
        setStatus(dom.detectStatus, "请先完成一次检测对照。", "error");
        return;
    }
    triggerDownload(state.currentDetectResult.compare_vis_url, `detect_vs_${Date.now()}.jpg`);
}

async function runCompare() {
    if (!state.compareFile) {
        setStatus(dom.compareStatus, "请先选择图片", "error");
        return;
    }
    try {
        dom.runCompare.disabled = true;
        setStatus(dom.compareStatus, "综合对比生成中...", "loading");
        const image = await fileToDataUrl(state.compareFile);
        const payload = await apiRequest("/api/compare", {
            filename: state.compareFile.name,
            image,
            traditional_key: dom.traditionalSelect.value,
            improved_key: dom.compareModelSelect.value,
        });
        renderComparisonCards(payload.result);
        setLink(dom.compareReportLink, payload.result.compare_report_url);
        renderBenchmarkHighlights(payload.result.benchmark_panels);
        renderBenchmarks(payload.result.benchmark_panels);
        const archiveResponse = await fetch("/api/archive");
        const archivePayload = await archiveResponse.json();
        state.bootstrap.archives = archivePayload.archives || [];
        renderArchiveCards(dom.archivePreviewGrid, archivePayload.archives || []);
        renderAchievementCards();
        setStatus(dom.compareStatus, "综合对比完成", "success");
        window.location.hash = "#compare";
        setActiveNavFromHash();
    } catch (error) {
        setStatus(dom.compareStatus, error.message, "error");
    } finally {
        dom.runCompare.disabled = false;
    }
}

function bindEvents() {
    dom.compareSlider.addEventListener("input", (event) => {
        updateCompareSlider(event.target.value);
    });
    dom.enhanceFile.addEventListener("change", (event) => {
        state.enhanceFile = event.target.files?.[0] || null;
        setStatus(dom.enhanceStatus, state.enhanceFile ? "图片已选择" : "等待上传", state.enhanceFile ? "success" : "");
        resetDetectPanel();
    });
    dom.compareFile.addEventListener("change", (event) => {
        state.compareFile = event.target.files?.[0] || null;
        setStatus(dom.compareStatus, state.compareFile ? "图片已选择" : "等待上传", state.compareFile ? "success" : "");
    });
    dom.runEnhance.addEventListener("click", runEnhance);
    dom.runCompare.addEventListener("click", runCompare);
    dom.runDetect.addEventListener("click", () => runDetect(false));
    dom.downloadDetectVis.addEventListener("click", downloadDetectVis);
    dom.detectConf.addEventListener("input", (event) => {
        dom.detectConfValue.textContent = Number(event.target.value).toFixed(2);
    });
    dom.downloadCurrentResult.addEventListener("click", downloadCurrentResult);
    dom.exportPresentationShot.addEventListener("click", exportPresentationShot);
    window.addEventListener("hashchange", setActiveNavFromHash);
    setActiveNavFromHash();
}

async function init() {
    updateCompareSlider(50);
    bindEvents();
    try {
        await bootstrap();
        enhancePremiumStyling();
        registerParallaxCards();
        registerRevealAnimations();
    } catch (error) {
        setStatus(dom.enhanceStatus, `初始化失败：${error.message}`, "error");
        setStatus(dom.compareStatus, `初始化失败：${error.message}`, "error");
    }
}

init();


