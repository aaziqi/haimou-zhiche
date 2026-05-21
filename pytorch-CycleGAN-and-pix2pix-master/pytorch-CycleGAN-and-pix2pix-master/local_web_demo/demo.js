const flowDom = {
    flowStart: document.getElementById("flowStart"),
    flowNext: document.getElementById("flowNext"),
    flowAuto: document.getElementById("flowAuto"),
    flowNarrative: document.getElementById("flowNarrative"),
    flowProgressBar: document.getElementById("flowProgressBar"),
    flowList: document.getElementById("flowList"),
    flowStepTitle: document.getElementById("flowStepTitle"),
    flowStepDesc: document.getElementById("flowStepDesc"),
    flowOpenPage: document.getElementById("flowOpenPage"),
    flowPreview: document.getElementById("flowPreview"),
};

const flowState = {
    currentIndex: -1,
    timer: null,
};

const steps = [
    {
        title: "首页封面区",
        desc: "先展示作品定位、封面海报区和核心成果卡片，建立第一印象。",
        url: "./index.html#overview",
        narrative: [
            "先从首页开场，突出作品名称、赛道定位和封面级展示效果。",
            "这一屏重点回答“你做的是什么作品”以及“为什么适合冲国奖”。",
        ],
    },
    {
        title: "智能增强演示",
        desc: "切到上传增强模块，展示图片增强前后的拖拽对比效果。",
        url: "./index.html#enhance",
        narrative: [
            "演示上传图片与模型切换，突出系统的交互性和可演示性。",
            "再用拖拽方式强调增强前后差异，让评委直观看到效果。",
        ],
    },
    {
        title: "综合对比页",
        desc: "说明传统方法、原始 CycleGAN 与改进模型的差异。",
        url: "./index.html#compare",
        narrative: [
            "把传统方法、原始 CycleGAN 和改进模型放在一起讲，最能体现方法演进价值。",
            "这一环节重点回答“为什么你的方案更优”。",
        ],
    },
    {
        title: "Benchmark 图表页",
        desc: "用可视化图表讲清楚客观指标、冠军标签和提升幅度。",
        url: "./benchmark.html",
        narrative: [
            "这一页是数据支撑页，强调指标可信度与竞赛表达完整性。",
            "图表中的冠军标签和提升箭头可帮助你快速口播结论。",
        ],
    },
    {
        title: "结果归档案例库",
        desc: "展示系统自动沉淀的案例库，体现作品持续积累能力。",
        url: "./archive.html",
        narrative: [
            "归档页说明这不是一次性演示，而是能持续积累案例的系统化作品。",
            "答辩结束前可回到这一页总结平台化价值与成果沉淀能力。",
        ],
    },
];

function renderNarrative(items) {
    flowDom.flowNarrative.innerHTML = "";
    items.forEach((item) => {
        const div = document.createElement("div");
        div.className = "bullet-item";
        div.textContent = item;
        flowDom.flowNarrative.appendChild(div);
    });
}

function renderStepList() {
    flowDom.flowList.innerHTML = "";
    steps.forEach((step, index) => {
        const article = document.createElement("article");
        article.className = "flow-step";
        article.innerHTML = `
            <div class="flow-step-index">${index + 1}</div>
            <h3>${step.title}</h3>
            <p>${step.desc}</p>
        `;
        article.addEventListener("click", () => activateStep(index));
        flowDom.flowList.appendChild(article);
    });
}

function setLink(href) {
    if (!href) {
        flowDom.flowOpenPage.href = "#";
        flowDom.flowOpenPage.classList.add("disabled");
        return;
    }
    flowDom.flowOpenPage.href = href;
    flowDom.flowOpenPage.classList.remove("disabled");
}

function activateStep(index) {
    flowState.currentIndex = index;
    const step = steps[index];
    Array.from(flowDom.flowList.children).forEach((item, itemIndex) => {
        item.classList.toggle("active", itemIndex === index);
    });
    flowDom.flowStepTitle.textContent = step.title;
    flowDom.flowStepDesc.textContent = step.desc;
    renderNarrative(step.narrative);
    flowDom.flowPreview.src = step.url;
    setLink(step.url);
    flowDom.flowProgressBar.style.width = `${((index + 1) / steps.length) * 100}%`;
}

function startFlow() {
    activateStep(0);
}

function nextFlow() {
    const nextIndex = flowState.currentIndex < steps.length - 1 ? flowState.currentIndex + 1 : 0;
    activateStep(nextIndex);
}

function toggleAutoPlay() {
    if (flowState.timer) {
        clearInterval(flowState.timer);
        flowState.timer = null;
        flowDom.flowAuto.textContent = "自动播放";
        return;
    }
    if (flowState.currentIndex === -1) {
        activateStep(0);
    }
    flowDom.flowAuto.textContent = "停止自动播放";
    flowState.timer = setInterval(() => {
        const nextIndex = flowState.currentIndex < steps.length - 1 ? flowState.currentIndex + 1 : 0;
        activateStep(nextIndex);
    }, 4500);
}

function initFlow() {
    renderStepList();
    renderNarrative([
        "点击“启动一键演示”后，系统会自动跳到首页第一步。",
        "你也可以手动点击左侧任一步骤，快速切换到对应展示页面。",
    ]);
    flowDom.flowStart.addEventListener("click", startFlow);
    flowDom.flowNext.addEventListener("click", nextFlow);
    flowDom.flowAuto.addEventListener("click", toggleAutoPlay);
}

initFlow();
