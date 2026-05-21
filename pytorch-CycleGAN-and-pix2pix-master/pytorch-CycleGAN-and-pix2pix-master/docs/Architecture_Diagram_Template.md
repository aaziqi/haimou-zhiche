# MP-CycleGAN 架构图模板

为了方便在论文中使用矢量图或更专业的绘图工具（如 Visio, PowerPoint, Draw.io）重绘架构图，以下提供了详细的**Mermaid**代码和绘图思路。

## 1. 架构逻辑流 (Mermaid Code)

你可以将以下代码复制到支持 Mermaid 的编辑器（如 Notion, Obsidian, GitHub）中查看效果。

```mermaid
graph LR
    subgraph Input
        RealA[输入水下图像\nReal A]
    end

    subgraph Generator_Cycle
        G_AB[生成器 G\n(A -> B)]
        FakeB[生成图像\nFake B]
        G_BA[生成器 F\n(B -> A)]
        RecA[重构图像\nRec A]
    end

    subgraph Discriminator
        RealB[真实清晰图像\nReal B]
        D_B[判别器 D_B]
    end

    subgraph Losses
        L_cyc((循环一致性损失\nL_cyc))
        L_gray((灰世界损失\nL_gray))
        L_ssim((结构相似性损失\nL_ssim))
        L_perc((感知损失\nL_perc))
        L_adv((对抗损失\nL_adv))
    end

    %% Data Flow
    RealA --> G_AB --> FakeB
    FakeB --> G_BA --> RecA
    
    %% Cycle Loss
    RealA -.-> L_cyc
    RecA -.-> L_cyc
    
    %% Discriminator Flow
    FakeB --> D_B
    RealB --> D_B
    D_B -.-> L_adv

    %% MP Losses (Our Contribution)
    FakeB -- 颜色校正 --> L_gray
    
    RealA -- 结构对比 --> L_ssim
    FakeB -- 结构对比 --> L_ssim
    
    RealA -- 特征提取(VGG) --> L_perc
    FakeB -- 特征提取(VGG) --> L_perc

    style RealA fill:#f9f,stroke:#333,stroke-width:2px
    style FakeB fill:#bbf,stroke:#333,stroke-width:2px
    style RecA fill:#dfd,stroke:#333,stroke-width:2px
    style RealB fill:#f9f,stroke:#333,stroke-width:2px
    style Losses fill:#fff,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
```

## 2. 绘图建议 (PowerPoint / Visio)

在 SCI 论文中，建议使用 PPT 或 Visio 绘制，导出为 PDF 或高分率 PNG (600 DPI)。

### 布局建议
1.  **左侧**：放置 `Real A (Underwater)` 输入。
2.  **中间上层**：放置正向生成路径 `G: A->B`，输出 `Fake B (Enhanced)`。
3.  **中间下层**：放置反向重构路径 `F: B->A`，输出 `Rec A`。
4.  **右侧**：放置 `Real B (Clear)` 和判别器 `D_B`。
5.  **关键损失标注 (Highlights)**：
    *   **Gray-World Loss**: 画一个箭头指向 `Fake B`，表示对生成结果的颜色约束。
    *   **SSIM / Perceptual Loss**: 画虚线连接 `Real A` 和 `Fake B`，表示两者在结构和内容上的一致性（这是非配对任务中的关键约束）。

### 配色方案
*   **生成器 (G, F)**: 浅蓝色背景
*   **判别器 (D)**: 浅橙色背景
*   **图像块**: 白色或浅灰色，带边框
*   **损失函数**: 用圆形或菱形表示，使用绿色或红色虚线连接。
