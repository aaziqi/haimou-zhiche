from pathlib import Path
repo = Path(r'd:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\local_web_demo')
app = repo / 'app.js'
text = app.read_text(encoding='utf-8-sig')
text = text.replace(
"""const state = {
    bootstrap: null,
    enhanceFile: null,
    compareFile: null,
    currentEnhanceResult: null,
    currentDetectResult: null,
    titleRotationTimer: null,
};""",
"""const state = {
    bootstrap: null,
    enhanceFile: null,
    compareFile: null,
    currentEnhanceResult: null,
    currentDetectResult: null,
    titleRotationTimer: null,
    runtimeMode: \"dynamic\",
};"""
)
text = text.replace(
"""    archivePreviewGrid: document.getElementById(\"archivePreviewGrid\"),
    presentationMeta: document.getElementById(\"presentationMeta\"),
    pptOutline: document.getElementById(\"pptOutline\"),
    demoScript: document.getElementById(\"demoScript\"),
};""",
"""    archivePreviewGrid: document.getElementById(\"archivePreviewGrid\"),
    presentationMeta: document.getElementById(\"presentationMeta\"),
    pptOutline: document.getElementById(\"pptOutline\"),
    demoScript: document.getElementById(\"demoScript\"),
    deployModeBanner: document.getElementById(\"deployModeBanner\"),
};"""
)
text = text.replace(
"""function setStatus(target, text, kind = \"\") {
    target.textContent = text;
    target.className = `status-chip${kind ? ` ${kind}` : \"\"}`;
}
""",
"""function setStatus(target, text, kind = \"\") {
    target.textContent = text;
    target.className = `status-chip${kind ? ` ${kind}` : \"\"}`;
}
function isStaticMode() {
    return state.runtimeMode === \"static\";
}
function showDeployModeBanner(message) {
    if (!dom.deployModeBanner) {
        return;
    }
    dom.deployModeBanner.hidden = false;
    dom.deployModeBanner.textContent = message;
}
"""
)
text = text.replace(
"""async function apiRequest(path, body) {
    const response = await fetch(path, {
        method: \"POST\",
        headers: {
            \"Content-Type\": \"application/json\",
        },
        body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || \"请求失败\");
    }
    return payload;
}
""",
"""async function apiRequest(path, body) {
    if (isStaticMode()) {
        throw new Error(\"当前为线上展示版，请在本地完整版中使用实时推理功能。\");
    }
    const response = await fetch(path, {
        method: \"POST\",
        headers: {
            \"Content-Type\": \"application/json\",
        },
        body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || \"请求失败\");
    }
    return payload;
}
"""
)
insert_block = """
function disableInteractiveRuntime() {
    showDeployModeBanner(\"当前为线上展示版：可查看作品介绍、图表、案例与报告；实时增强、综合对比和检测功能仅在本地完整版开放。\");
    if (dom.runEnhance) {
        dom.runEnhance.disabled = true;
        dom.runEnhance.textContent = \"线上展示版不开放实时增强\";
    }
    if (dom.runCompare) {
        dom.runCompare.disabled = true;
        dom.runCompare.textContent = \"线上展示版不开放实时对比\";
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
    setStatus(dom.enhanceStatus, \"线上展示版：请查看预生成案例与报告\", \"loading\");
    setStatus(dom.compareStatus, \"线上展示版：请查看预生成综合对比案例\", \"loading\");
    setStatus(dom.detectStatus, \"线上展示版：检测对照仅在本地完整版开放\", \"loading\");
    resetDetectPanel(\"线上展示版不提供实时检测，请查看归档中的对照案例。\");
}
function hydrateStaticPreview() {
    const enhanceEntry = state.bootstrap?.archives?.find((item) => item.type === \"enhance\");
    const compareEntry = state.bootstrap?.archives?.find((item) => item.type === \"compare\");
    const detectEntry = state.bootstrap?.archives?.find((item) => item.type === \"detect_compare\");
    if (enhanceEntry) {
        renderEnhanceResult({
            input_url: enhanceEntry.input_url,
            output_url: enhanceEntry.cover_url,
            report_url: enhanceEntry.report_url,
            metrics_url: enhanceEntry.metrics_url,
            processing_seconds: \"预生成案例\",
            key: enhanceEntry.id,
            output_uciqe: null,
            output_uiqm: null,
            delta_uciqe: null,
            delta_uiqm: null,
            model: {
                label: enhanceEntry.model_label || \"改进模型 MPCGAN\",
                description: \"线上展示版使用预生成增强案例，实时推理功能保留在本地完整版。\",
                checkpoint_name: \"euvp_mpcgan_stage2_s0\",
                epoch: \"202\",
            },
        });
    }
    if (compareEntry) {
        const classic = state.bootstrap?.models?.find((item) => item.key === \"classic_cyclegan\");
        const improved = state.bootstrap?.models?.find((item) => item.key === \"improved_mpcgan\");
        renderComparisonCards({
            input_url: compareEntry.input_url,
            variants: [
                {
                    badge: \"传统方法\",
                    label: \"传统方法 Gray World + CLAHE\",
                    description: \"作为基础增强方案，适合展示颜色校正与局部对比度提升效果。\",
                    image_url: compareEntry.input_url,
                    metrics: {},
                    report_url: \"\",
                },
                {
                    badge: classic?.badge || \"基线模型\",
                    label: classic?.label || \"原 CycleGAN\",
                    description: classic?.description || \"无配对图像翻译基线模型。\",
                    image_url: compareEntry.cover_url,
                    metrics: {},
                    report_url: \"\",
                },
                {
                    badge: improved?.badge || \"主推模型\",
                    label: improved?.label || compareEntry.model_label || \"改进模型 MPCGAN\",
                    description: improved?.description || \"当前适合作为主推方案的增强模型。\",
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
                label: detectEntry.model_label || \"下游检测器 (YOLO)\",
            },
            report_url: detectEntry.report_url,
            raw_json_url: detectEntry.extra?.raw_json_url || \"\",
            enhanced_json_url: detectEntry.extra?.enhanced_json_url || detectEntry.metrics_url || \"\",
            conf: Number(dom.detectConf?.value ?? 0.25),
        });
    }
}
"""
text = text.replace("async function bootstrap() {", insert_block + "async function bootstrap() {")
text = text.replace(
"""async function bootstrap() {
    const response = await fetch(\"/api/bootstrap\");
    const payload = await response.json();
    state.bootstrap = payload;""",
"""async function bootstrap() {
    let payload;
    try {
        const response = await fetch(\"/api/bootstrap\", { cache: \"no-store\" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        payload = await response.json();
        state.runtimeMode = payload.runtime_mode || \"dynamic\";
    } catch (error) {
        const response = await fetch(\"./data/bootstrap.json\", { cache: \"no-store\" });
        if (!response.ok) {
            throw new Error(\"无法读取展示数据，请检查 data/bootstrap.json 是否存在。\");
        }
        payload = await response.json();
        state.runtimeMode = \"static\";
    }
    state.bootstrap = payload;"""
)
text = text.replace(
"""    renderBenchmarkHighlights(payload.benchmarks);
    renderBenchmarks(payload.benchmarks);
    renderPresentationPack(payload.presentation_pack);
    renderArchiveCards(dom.archivePreviewGrid, payload.archives || []);
}""",
"""    renderBenchmarkHighlights(payload.benchmarks);
    renderBenchmarks(payload.benchmarks);
    renderPresentationPack(payload.presentation_pack);
    renderArchiveCards(dom.archivePreviewGrid, payload.archives || []);
    if (isStaticMode()) {
        disableInteractiveRuntime();
        hydrateStaticPreview();
    }
}"""
)
app.write_text(text, encoding='utf-8')
print('app updated')
