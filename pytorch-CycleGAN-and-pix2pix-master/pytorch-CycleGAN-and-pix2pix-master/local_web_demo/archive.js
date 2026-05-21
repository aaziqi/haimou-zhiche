const archiveDom = {
    archiveStory: document.getElementById("archiveStory"),
    archiveSummary: document.getElementById("archiveSummary"),
    archiveStatsGrid: document.getElementById("archiveStatsGrid"),
    archiveTagChips: document.getElementById("archiveTagChips"),
    archiveSearch: document.getElementById("archiveSearch"),
    archiveTypeFilter: document.getElementById("archiveTypeFilter"),
    archiveModelFilter: document.getElementById("archiveModelFilter"),
    archiveSort: document.getElementById("archiveSort"),
    archiveFilterSummary: document.getElementById("archiveFilterSummary"),
    archiveReset: document.getElementById("archiveReset"),
    archivePinnedGrid: document.getElementById("archivePinnedGrid"),
    archiveLibrary: document.getElementById("archiveLibrary"),
};
const archiveState = {
    entries: [],
    selectedTag: "all",
    expanded: new Set(),
};
function renderArchiveBullets(items) {
    archiveDom.archiveStory.innerHTML = "";
    items.forEach((item) => {
        const div = document.createElement("div");
        div.className = "bullet-item";
        div.textContent = item;
        archiveDom.archiveStory.appendChild(div);
    });
}
function renderArchiveSummary(entries) {
    const enhanceCount = entries.filter((item) => item.type === "enhance").length;
    const compareCount = entries.filter((item) => item.type === "compare").length;
    const detectCount = entries.filter((item) => item.type === "detect_compare").length;
    archiveDom.archiveSummary.innerHTML = `
        <span class="metric-pill">总案例 ${entries.length}</span>
        <span class="metric-pill">单图增强 ${enhanceCount}</span>
        <span class="metric-pill">综合对比 ${compareCount}</span>
        <span class="metric-pill">检测对照 ${detectCount}</span>
    `;
}
function priorityScore(entry, index) {
    let score = Math.max(0, 100 - index);
    if ((entry.tags || []).includes("主推模型")) {
        score += 40;
    }
    if (entry.type === "compare") {
        score += 24;
    }
    if ((entry.tags || []).includes("综合对比")) {
        score += 18;
    }
    return score;
}
function renderArchiveStats(entries) {
    const models = new Set(entries.map((item) => item.model_label).filter(Boolean)).size;
    const tags = new Set(entries.flatMap((item) => item.tags || [])).size;
    const compareCount = entries.filter((item) => item.type === "compare").length;
    const enhanceCount = entries.filter((item) => item.type === "enhance").length;
    archiveDom.archiveStatsGrid.innerHTML = `
        <article class="archive-stat-card"><span>归档总量</span><strong>${entries.length}</strong></article>
        <article class="archive-stat-card"><span>模型覆盖</span><strong>${models}</strong></article>
        <article class="archive-stat-card"><span>单图增强</span><strong>${enhanceCount}</strong></article>
        <article class="archive-stat-card"><span>综合对比</span><strong>${compareCount}</strong></article>
    `;
    if (tags > 0) {
        archiveDom.archiveSummary.innerHTML += `<span class="metric-pill">标签 ${tags}</span>`;
    }
}
function renderTagChips(entries) {
    const tags = Array.from(new Set(entries.flatMap((item) => item.tags || [])));
    archiveDom.archiveTagChips.innerHTML = "";
    const allTags = ["all", ...tags];
    allTags.forEach((tag) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `tag-chip${archiveState.selectedTag === tag ? " active" : ""}`;
        button.textContent = tag === "all" ? "全部标签" : tag;
        button.addEventListener("click", () => {
            archiveState.selectedTag = tag;
            renderTagChips(entries);
            filterArchiveEntries();
        });
        archiveDom.archiveTagChips.appendChild(button);
    });
}
function createArchiveCard(entry, pinned = false) {
    const article = document.createElement("article");
    article.className = `archive-card${archiveState.expanded.has(entry.id) ? " expanded" : ""}`;
    const tags = (entry.tags || []).map((tag) => `<span class="micro-badge">${tag}</span>`).join("");
    article.innerHTML = `
        ${pinned ? '<div class="archive-pin">重点案例</div>' : ""}
        <img src="${entry.cover_url}" alt="${entry.title}">
        <div class="archive-meta"><span>${entry.created_at}</span><span>${entry.model_label || "案例"}</span></div>
        <h3>${entry.title}</h3>
        <p>${entry.summary}</p>
        <div class="badge-row">${tags}</div>
        <div class="archive-actions">
            <button class="ghost-button archive-expand-btn" type="button">${archiveState.expanded.has(entry.id) ? "收起详情" : "展开详情"}</button>
            <a class="ghost-link" href="${entry.report_url}" target="_blank" rel="noopener">打开案例报告</a>
            ${entry.metrics_url ? `<a class="ghost-link" href="${entry.metrics_url}" target="_blank" rel="noopener">打开明细入口</a>` : ""}
        </div>
        <div class="archive-detail">
            <div class="meta-line">案例类型：${entry.type}</div>
            <div class="meta-line">模型名称：${entry.model_label || "未标注"}</div>
            <div class="meta-line">归档编号：${entry.id}</div>
            <div class="meta-line">封面地址：${entry.cover_url}</div>
        </div>
    `;
    article.querySelector(".archive-expand-btn").addEventListener("click", () => {
        if (archiveState.expanded.has(entry.id)) {
            archiveState.expanded.delete(entry.id);
        } else {
            archiveState.expanded.add(entry.id);
        }
        filterArchiveEntries();
    });
    return article;
}
function renderPinnedEntries(entries) {
    archiveDom.archivePinnedGrid.innerHTML = "";
    if (!entries.length) {
        const empty = document.createElement("div");
        empty.className = "bullet-item";
        empty.textContent = "当前没有可置顶的重点案例。";
        archiveDom.archivePinnedGrid.appendChild(empty);
        return;
    }
    entries.slice(0, 3).forEach((entry) => {
        archiveDom.archivePinnedGrid.appendChild(createArchiveCard(entry, true));
    });
}
function renderArchiveLibrary(entries) {
    archiveDom.archiveLibrary.innerHTML = "";
    if (!entries.length) {
        const empty = document.createElement("div");
        empty.className = "bullet-item";
        empty.textContent = "暂无归档案例。";
        archiveDom.archiveLibrary.appendChild(empty);
        return;
    }
    entries.forEach((entry) => {
        archiveDom.archiveLibrary.appendChild(createArchiveCard(entry, false));
    });
}
function renderFilterOptions(entries) {
    const types = Array.from(new Set(entries.map((item) => item.type))).filter(Boolean);
    const models = Array.from(new Set(entries.map((item) => item.model_label))).filter(Boolean);
    archiveDom.archiveTypeFilter.innerHTML = '<option value="all">全部类型</option>' + types.map((type) => `<option value="${type}">${type}</option>`).join("");
    archiveDom.archiveModelFilter.innerHTML = '<option value="all">全部模型</option>' + models.map((model) => `<option value="${model}">${model}</option>`).join("");
}
function filterArchiveEntries() {
    const keyword = archiveDom.archiveSearch.value.trim().toLowerCase();
    const type = archiveDom.archiveTypeFilter.value;
    const model = archiveDom.archiveModelFilter.value;
    const sort = archiveDom.archiveSort.value;
    let entries = archiveState.entries.filter((entry) => {
        const text = `${entry.title} ${entry.summary} ${entry.model_label || ""}`.toLowerCase();
        const keywordMatch = !keyword || text.includes(keyword);
        const typeMatch = type === "all" || entry.type === type;
        const modelMatch = model === "all" || entry.model_label === model;
        const tagMatch = archiveState.selectedTag === "all" || (entry.tags || []).includes(archiveState.selectedTag);
        return keywordMatch && typeMatch && modelMatch && tagMatch;
    });
    if (sort === "oldest") {
        entries = [...entries].reverse();
    } else if (sort === "model") {
        entries = [...entries].sort((a, b) => (a.model_label || "").localeCompare(b.model_label || "", "zh-CN"));
    }
    archiveDom.archiveFilterSummary.textContent = `当前显示 ${entries.length} / ${archiveState.entries.length} 个案例`;
    const pinned = [...entries]
        .map((entry) => ({ entry, score: priorityScore(entry, archiveState.entries.indexOf(entry)) }))
        .sort((a, b) => b.score - a.score)
        .map((item) => item.entry);
    renderPinnedEntries(pinned);
    renderArchiveLibrary(entries);
}
function bindArchiveFilters() {
    [archiveDom.archiveSearch, archiveDom.archiveTypeFilter, archiveDom.archiveModelFilter, archiveDom.archiveSort].forEach((element) => {
        element.addEventListener("input", filterArchiveEntries);
        element.addEventListener("change", filterArchiveEntries);
    });
    archiveDom.archiveReset.addEventListener("click", () => {
        archiveDom.archiveSearch.value = "";
        archiveDom.archiveTypeFilter.value = "all";
        archiveDom.archiveModelFilter.value = "all";
        archiveDom.archiveSort.value = "newest";
        filterArchiveEntries();
    });
}
async function loadArchives() {
    try {
        const response = await fetch("/api/archive", { cache: "no-store" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        const response = await fetch("./data/archive.json", { cache: "no-store" });
        if (!response.ok) {
            throw new Error("无法读取归档展示数据。");
        }
        return await response.json();
    }
}
async function initArchive() {
    renderArchiveBullets([
        "每次单图增强后，系统会自动保存增强结果案例，便于筛选最优展示样本。",
        "每次综合对比后，系统会自动保存对比案例，便于构建完整的软件应用展示故事线。",
        "建议展示前挑选最具代表性的海报型样例，形成固定讲解脚本与成果集。",
    ]);
    const payload = await loadArchives();
    archiveState.entries = payload.archives || [];
    renderArchiveSummary(archiveState.entries);
    renderArchiveStats(archiveState.entries);
    renderFilterOptions(archiveState.entries);
    renderTagChips(archiveState.entries);
    bindArchiveFilters();
    filterArchiveEntries();
}
initArchive().catch((error) => {
    renderArchiveBullets([`归档页初始化失败：${error.message}`]);
});
