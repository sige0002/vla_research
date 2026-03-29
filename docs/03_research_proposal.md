# 研究提案書

> Qwen3.5-4Bを軸としたVLA研究の方向性、研究要素、実装計画を統合的にまとめる。
> 旧 01_team_discussion.md, 02_research_elements.md, 03_implementation_order.md を統合。

---

## 1. 研究の位置づけ

### 1.1 出発点（VLA_idea.md）

```
High Policy (5Hz):
  入力: 短動画 + タスク目標 + 前回の長期言語メモリ
  バックボーン: Qwen3.5-4B (ビデオ対応VLM)
  出力: 現在のサブタスク + 更新された長期言語メモリ

Low Policy (30-100Hz):
  入力: サブタスク指示 + 現在の観測
  出力: ロボット行動
```

### 1.2 目標

- 成功率が高い（LIBERO 98%+、実世界で実用レベル）
- 研究新規性がありSOTA狙い（CoRL/RSS/ICLR投稿レベル）
- ロングタスク対応（15分級）
- 軽量RL（RLT等）による成功率改善パイプライン
- 高ポリシー5Hz、低ポリシー30Hz以上（理想100Hz）

### 1.3 主要参考文献（基盤技術）

| 技術 | 文献 |
|------|------|
| **Qwen3.5** | Qwen Team (2025). "Qwen3.5 Technical Report." — Gated DeltaNet hybrid attention, DeepStack廃止 |
| **Gated DeltaNet** | Yang, S. et al. (2024). "Gated Delta Networks: Improving Mamba2 with Delta Rule." *arXiv:2412.06464* |
| **DeltaNet** | Yang, S. et al. (2024). "Parallelizing Linear Transformers with the Delta Rule over Sequence Length." *ICML 2024* |
| **MEM** | Physical Intelligence (2025). "MEM: Memory-Enhanced Manipulation." — 短期ビデオメモリ+長期言語メモリによるロングタスク対応 |
| **RLT** | Physical Intelligence (2025). "Reinforcement Learning Tokens." — VLA埋め込みからRLトークン抽出、軽量Actor-Criticで行動精密編集 |
| **pi0 / pi0.5** | Black, K. et al. (2024). "π0: A Vision-Language-Action Flow Model for General Robot Control." *arXiv:2410.24164* |
| **OpenVLA** | Kim, M.J. et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action Model." *arXiv:2406.09246* |
| **Flow Matching** | Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." *ICLR 2023* |
| **FAST** | Pertsch, K. et al. (2025). "FAST: Efficient Action Tokenization for Vision-Language-Action Models." *RSS 2025* |
| **LIBERO** | Liu, B. et al. (2023). "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." *NeurIPS 2023* |
| **StarVLA** | 実装フレームワーク（Qwen-FAST/Qwen-PI対応） |

---

## 2. 重要な技術的制約（調査で判明）

### Qwen3.5のViTにはフレーム間attentionがない

Qwen開発チーム確認済み: ViT内の時間融合は入力段3D Convのみ。全フレームの全トークンがLLMに渡される。

→ **MEM的改造（ViTへの時間attention追加 + 過去フレームドロップ）は必須。**
→ ただしQwen3.5はDeepStack廃止済みなので改造はシンプルに実装可能。

詳細は `02_qwen35_backbone.md` セクション2を参照。

---

## 3. 方向性の検討

調査の結果、大きく4つの研究方向が候補に挙がる。

| 方向 | 概要 | 利点 | リスク |
|------|------|------|--------|
| **A: PI技術統合** (MEM + RLT) | PIが個別発表した技術をQwen3.5上で統合 | 各技術の有効性は実証済み、ロングタスク+精密制御 | PI技術の再実装の域を出ない恐れ、詳細がクローズド |
| **B: 離散拡散 + テスト時適応** | DIVA/DD-VLAの並列アクション生成 + RD-VLAの再帰精製 | PIのFMと明確に差別化、LIBERO 97.4%実績 | 離散拡散VLAの実装難度高、MTPとの相互作用未知 |
| **C: ワールドモデル共進化** | VLAWのポリシー-WM共進化 + LRMの報酬自動生成 | 自律改善ループが研究として強い、VLAW +39.2% | WM学習のデータ・計算コスト大、sim-to-realギャップ |
| **D: Embodied CoT + 3D空間推論** | HALOのEM-CoT + SpatialVLA/Any3D-VLAの空間推論 | Qwen3.5の推論能力を最大化 | CoTオーバーヘッド、3Dセンサ依存 |

### pretrain観点での比較評価

| 評価軸 | A (PI統合) | B (離散拡散) | C (WM共進化) | D (Embodied CoT) |
|--------|-----------|-------------|-------------|-------------------|
| pretrain設計への影響 | ViT改造必須、初期から時間attention | 標準的 | WM同時学習が理想だが高コスト | VLM既存能力を活用、軽い |
| DeltaNetとの親和性 | ★★★★ 再帰状態→メモリ活用 | ★★★ 直交 | ★★★ 直交 | ★★★★★ 高速推論がCoT相殺 |
| 研究新規性（pretrain） | △ MEM再現リスク | △ なし | ○ WM共学習は新しい | △ なし |
| 認知発達との整合性 | ★★★★★ 段階的メモリ獲得 | ★★ 発達後期の話 | ★★★ 内部モデル構築 | ★★★★ 高次認知機能 |

### 方向性の結論

**方向Aをベースに認知発達的カリキュラムで差別化。** 理由:

1. **pretrainの本質的問題に最も近い**: 方向B/Dはpretrain設計の革新に繋がらない。方向Aはpretrain初期からViT改造が必要で、設計全体を再考する機会を与える
2. **DeltaNetの活用がpretrain段階で最も差別化**: Gated DeltaNetの再帰状態のVLAでの振る舞いは**完全に未知の領域**であり、それ自体が研究貢献
3. **認知発達的カリキュラムとの統合が自然**: メモリ獲得過程が乳児の作業記憶発達と対応。PI技術の再実装と明確に差別化できる
4. **方向B/C/Dの要素はposttrainで統合可能**: 離散拡散→finetune段階、WM共進化→RL段階、CoT→High Policy推論強化として後段で取り込む

### 推奨構成

1. **認知発達的カリキュラムpretrain**: フレーム数・タスク複雑度・言語条件の3軸を段階的に拡張（アイデア6）
2. **Gated DeltaNet分析**: 再帰状態のメモリ的振る舞いの分析をpretrain段階で実施
3. **アフォーダンス表現学習**: ViTのpretrain補助損失として導入（アイデア7）
4. **模倣の段階性**: 視覚模倣 → 意図理解 → 言語条件付けの順序（アイデア8）
5. **アクション生成方式**: pretrain後にfinetune段階で選定（方向Bから離散拡散 or ActionCodec）
6. **RL改善**: RLTを中心とした軽量RLパイプライン（後段、セクション5参照）

### pretrainの3つの重要決定事項

1. **カリキュラム設計**: 段階的 vs 一括。認知発達的カリキュラム（段階的）を推奨
2. **ViT temporal attentionの学習タイミング**: MEM論文の知見（事前学習なしで45%低下）から初期導入を推奨
3. **言語能力維持戦略**: Phase 1-Aで言語ヘッド凍結 + 後期にrehearsal混合を推奨

---

## 4. 提案アーキテクチャ

```
┌──────────────────────────────────────────────────┐
│              High-Level Policy (5Hz)              │
│                                                   │
│  Video ──► Qwen3.5 ViT ──► Spatial Merge          │
│  (K frames)   + MEM時間アテンション                │
│                    │                               │
│  Task goal ────────┤                               │
│  Memory m_t ───────┤                               │
│                    ▼                               │
│           Qwen3.5 LLM (32層 Hybrid)               │
│           [DeltaNet×3 + Full×1] × 8               │
│           + AWR/DPO条件付け (optional)             │
│                    │                               │
│              ┌─────┴─────┐                         │
│         subtask l    memory m_{t+1}                │
├──────────────┼─────────────────────────────────────┤
│              │   Low-Level Policy (30-100Hz)       │
│              ▼                                     │
│  Current obs ──► Action Expert                     │
│  Proprio ──────► (DiT FM / 離散拡散 / ActionCodec) │
│                        │                           │
│                  action chunk (10-20 steps)         │
│                  + RLT精密編集                      │
└────────────────────────────────────────────────────┘
```

---

## 5. RL改善パイプライン: RECAP vs RLT の分析と設計

### 5.1 RECAP + RLT 共存性の分析

RECAPとRLTは**階層的に分離された異なるレベルで動作**し、共存可能である。

| 観点 | RECAP | RLT |
|------|-------|-----|
| **動作レベル** | High Policy（サブタスク生成の品質改善） | Low Policy（アクション精密編集） |
| **メカニズム** | アドバンテージ条件付け + 分類器フリーガイダンス(CFG) | VLA埋め込みからRLトークン抽出 → 小規模Actor-Critic |
| **改善対象** | 戦略的判断（何をするか） | 戦術的実行（どう実行するか） |
| **学習コスト** | 中（データ収集→アドバンテージ推定→VLM再学習の数日サイクル） | 低（オンラインRL、15分〜2時間で収束） |
| **推論コスト** | **高: VLM 2回フォワードパス（CFGのため）** | **極低: MLP 1回（<1ms）** |
| **勾配干渉** | なし（階層分離） | なし（VLAバックボーン凍結） |

### 5.2 速度面の問題: RECAPのCFGがボトルネック

RECAPのCFG（分類器フリーガイダンス）は推論時にVLMの**2回フォワードパス**を要求する:

```
π*(a|o,l) ∝ π_ref(a|o,l) × (π_ref(a|I,o,l) / π_ref(a|o,l))^β
                              ↑条件付き        ↑無条件
                              1回目のpass      2回目のpass
```

Qwen3.5-4Bの1回パスが~150-200ms → 2回で~300-400ms → **5Hz（200ms）を超過**。

一方RLTは小規模MLPの1回パスのみで<1ms。30-100Hzに影響なし。

### 5.3 推奨: RLTベースの軽量RLパイプライン

速度制約を考慮し、**RECAPの完全なCFG方式は採用せず**、以下の軽量RL構成を推奨:

#### Low Policy: RLT（メイン）

RLTをLow Policyのアクション精密編集に採用。PI方式をベースにQwen3.5のDeltaNet埋め込みを活用。

**推論パイプライン**:
1. Action Expert (DiT FM等) が action chunk を生成
2. RLトークンエンコーダが VLA内部埋め込みから圧縮トークンを抽出
3. 小規模Actor-Criticが action correction δa を出力
4. 最終行動 = action_chunk[i] + δa_i

**学習**: オンラインRL（PPO/SAC）、VLAバックボーン凍結、15分〜2時間で収束。

#### High Policy: AWR（RECAPの軽量代替）

RECAPのCFGの代わりに、**Advantage-Weighted Regression (AWR)** を採用:

```
L = -E[ max(0, A(s,a)) × log π(subtask | obs, goal, mem) ]
```

AWRの利点:
- **推論コスト追加ゼロ**: CFGの2回フォワードパスが不要。通常の1回パスのみ
- **実装がシンプル**: アドバンテージ推定 → 重み付きSFTで再学習するだけ
- **RECAPの本質的メリットは保持**: 良いサブタスク分解に偏った学習ができる

AWRで不十分な場合のフォールバック: DPO（Direct Preference Optimization）を適用。これもCFG不要で推論コスト追加ゼロ。

### 5.4 軽量RL手法の比較（文献付き）

| 手法 | 推論コスト | 学習コスト | リアルタイム性 | 適用レベル | 文献 |
|------|-----------|-----------|-------------|-----------|------|
| **RLT** | <1ms (MLP) | 低 (小AC) | ★★★★★ 100Hz+ | Low | Pertsch et al. (2025). "Reinforcement Learning Tokens." Physical Intelligence |
| **Residual RL** | <1ms (MLP) | 低 (SAC/TD3) | ★★★★★ 100Hz+ | Low | Johannink et al. (2019). "Residual Reinforcement Learning for Robot Manipulators." *ICRA 2019*; Silver et al. (2018). "Residual Policy Learning." *arXiv:1812.06298* |
| **AWR** | 0 (通常推論) | 低 (重み付きSFT) | ★★★★★ | High | Peng, X.B. et al. (2019). "Advantage-Weighted Regression: Simple and Scalable Off-Policy RL." *arXiv:1910.00177* |
| **DPO** | 0 (通常推論) | 低 (ペア損失) | ★★★★★ | High | Rafailov et al. (2023). "Direct Preference Optimization." *NeurIPS 2023* |
| **VLA-RL (LoRA)** | 0 (同一モデル) | 中 (LoRA逆伝播) | ★★★★ 推論のみ | Both | Zhan, A. et al. (2025). "You Can't Spell VLA without RL." *arXiv preprint* |
| **SERL** | 依存 | 中 (フルパイプライン) | ★★★ 学習は非リアルタイム | Low | Luo, J. et al. (2024). "SERL: A Software Suite for Sample-Efficient Robotic RL." *RSS 2024* |
| **REBEL** | 0 (通常推論) | 低 (回帰) | ★★★★★ | High | Gao, L. et al. (2024). "REBEL: Reward-Based Regression for Efficient LM Training." *arXiv preprint* |
| **RECAP (CFG)** | **高 (2×VLM)** | 中 (条件付き再学習) | ★★ 5Hz超過リスク | High | Physical Intelligence (2025). "RECAP." |

### 5.5 推奨RL構成まとめ

```
High Policy (5Hz)
  └─ AWR or DPO（推論コスト追加ゼロ、オフライン学習）
       ├─ サブタスク分解品質の改善
       └─ 報酬: LRM（VLMベース報酬モデル）でサブタスク成功を判定

Low Policy (30-100Hz)
  └─ RLT（<1ms追加、オンライン学習15分〜2時間）
       ├─ アクション精密編集
       └─ 報酬: タスク報酬（ManiSkillでの予備検証 → 実機）
```

**RECAPのCFGは不採用**。AWR/DPOで同等の戦略改善を推論コストゼロで達成する。RLTの実機学習速度（15分〜2時間）は実用上十分であり、重いオフラインRLサイクルを回す必要がない。

**Residual RLはRLTの簡易版として検討可能**: RLトークン抽出の学習が困難な場合、状態ベースの残差ポリシー（MLP 2-3層）で代替。精度は下がるが実装が容易で100Hz+動作。

---

## 6. 研究要素と新規性

### 6.1 確実に主張できる新規性

| 要素 | 新規性の根拠 |
|------|------------|
| **Qwen3.5-4BベースVLA** | Gated DeltaNet（線形アテンション）を持つVLMのVLA応用は未発表 |
| **DeltaNet再帰状態のメモリ活用** | 固定サイズ再帰状態を暗黙的ワーキングメモリとして分析・活用 |
| **MEM on Qwen3.5** | MEM（Gemma 3ベース）のQwen3.5移植。DeepStack廃止による簡素化 |
| **ViT時間attention改造** | Qwen3.5 ViTへのMEM式temporal attention移植（改造必須と判明） |
| **認知発達的カリキュラムpretrain** | 感覚運動段階理論に基づくVLAのpretrain設計は完全に未探索 |

### 6.2 技術的差別化要素（既存VLA手法から）

| 要素 | 出典 | 優先度 | 備考 |
|------|------|--------|------|
| 離散拡散アクション生成 | DIVA/DD-VLA | 高 | finetune段階で検討 |
| テスト時再帰精製 | RD-VLA | 中 | Phase 4で検討 |
| VLMベース報酬モデル | LRM | 高 | AWR/DPOの報酬信号として使用 |
| ActionCodec (RVQトークナイザ) | ActionCodec | 中 | FAST代替として検討 |
| 3D空間推論 | SpatialVLA Ego3D PE | 低 | 低コストで追加可能だが優先度低 |

### 6.3 認知科学・神経科学からの研究アイデア（アーキテクチャに直結するもの）

#### アイデア2: ワーキングメモリゲーティングVLA【採用予定】

**着想**: PFC-基底核の動的ゲーティング — どの情報を保持/更新/破棄するかを学習する神経メカニズム

**主要文献**:
- O'Reilly & Frank (2006). "Making Working Memory Work." *Neural Computation*, 18(2), 283-328.
- Hazy et al. (2007). "Towards an Executive Without a Homunculus." *Phil. Trans. R. Soc. B*, 362(1485), 1601-1613.
- Graves et al. (2014). "Neural Turing Machines." *arXiv:1410.5401*

**VLAへの適用**: MEMの「成功時のみ更新」ヒューリスティックを、学習可能なゲーティングネットワークに置換。DeltaNet層 × 3 + Full層 × 1の構成がPFC-BGの「ゲート → バッファ → 更新」ループと構造的に類似しており、この対応関係を活用。

### 6.4 認知発達からの研究アイデア【pretrain設計の中核】

#### アイデア6: 発達的カリキュラム学習【採用予定】

**主要文献**:
- Piaget, J. (1952). *The Origins of Intelligence in Children*. — 感覚運動期の6段階
- Elman, J.L. (1993). "Learning and development in neural networks: the importance of starting small." *Cognition*, 48(1), 71-99. — 初期に入力を制限すると最終性能が向上
- Bengio, Y. et al. (2009). "Curriculum Learning." *ICML 2009*. — MLでのカリキュラム学習の定式化

**適用**: pretrainをPiagetの感覚運動段階に対応させた3サブフェーズ（Phase 1-A/B/C）に分割。詳細はVLA_idea.md Stage 1修正案を参照。

#### アイデア7: アフォーダンス知覚の発達的獲得【採用予定】

**主要文献**:
- Gibson, J.J. (1979). *The Ecological Approach to Visual Perception*. — アフォーダンス概念の原著
- Gibson, E.J. & Pick, A.D. (2000). *An Ecological Approach to Perceptual Learning and Development*. — 知覚学習の発達的メカニズム

**適用**: ViTのpretrain補助損失としてアフォーダンス予測を導入。行動データから自動ラベリング可能（成功把持位置 = graspable affordance）。

#### アイデア8: 社会的学習と模倣の段階性【採用予定】

**主要文献**:
- Meltzoff, A.N. (1988). "Infant imitation after a 1-week delay." *Developmental Psychology*, 24(4), 470-476.
- Tomasello, M. (1999). *The Cultural Origins of Human Cognition*. — 模倣→意図理解→教示の段階性

**適用**: pretrain段階で視覚模倣 → 意図理解 → 言語条件付けと段階的にタスク抽象度を上げる。VLMの言語能力を温存しつつロボットドメインに接地。

#### アイデア9: 対象永続性と予測的処理【採用予定】

**主要文献**:
- Baillargeon, R. (1987). "Object permanence in 3½- and 4½-month-old infants." *Developmental Psychology*, 23(5), 655-664.
- Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science." *BBS*, 36(3), 181-204.
- Rao & Ballard (1999). "Predictive coding in the visual cortex." *Nature Neuroscience*, 2(1), 79-87.

**適用**: pretrainの補助タスクとして遮蔽予測を導入。DeltaNetの再帰状態が遮蔽中の物体表象を保持するかはQwen3.5固有の検証ポイント。

### 6.5 その他の認知科学アイデア（将来の拡張候補）

以下は直接の採用は見送るが、将来の発展方向として記録する:

| アイデア | 着想 | 主要文献 | 見送り理由 |
|---------|------|---------|-----------|
| **睡眠固定化VLA** (1) | 海馬リプレイによる記憶転送 | Diekelmann & Born (2010) *Nat. Rev. Neurosci.* 11(2); Robins (1995) *Connection Science* 7(2) | 二相サイクルの計算コスト・スケジューリングが複雑。カリキュラム間のrehearsal混合で代替可能 |
| **三重学習系VLA** (3) | 小脳-基底核-皮質の協調 | Doya (1999) *Neural Networks* 12(7-8); Wolpert et al. (1998) *TiCS* 2(9) | β振動の計算モデル化が非自明。段階的構成（VLM→小脳→基底核）で部分的に実現可能 |
| **Hopfield連想記憶** (4) | Modern Hopfield NetworkとTransformerの等価性 | Ramsauer et al. (2021) *ICLR 2021* | pretrainに影響軽微。後段のアクション生成方式として検討する際に再評価 |
| **能動推論VLA** (5) | 自由エネルギー原理 | Friston (2010) *Nat. Rev. Neurosci.* 11(2) | 変分推論の計算コストが5Hzリアルタイム制約と矛盾。確信度スコア（precision weighting）のみ部分採用 |

---

## 7. 実装計画

### Phase 0: 環境構築（1週間）

評価環境: LIBERO, SimplerEnv, CALVIN。StarVLAフレームワークの動作確認。

### Phase 1: ベースVLA + 認知発達的カリキュラムpretrain（2-3週間）

**目標**: Qwen3.5-4B + アクションヘッドでLIBERO 90%+

3サブフェーズ構成（詳細はVLA_idea.md Stage 1修正案）:
- **Phase 1-A**: 視覚-運動接地（6フレーム、言語ヘッド凍結、アフォーダンス補助損失）
- **Phase 1-B**: サブタスク分解の獲得（12フレーム、言語段階的解放）
- **Phase 1-C**: メモリ管理の精緻化（18フレーム、ゲーティングネットワーク導入）

**判断ポイント**: Phase 1-C完了後にアクション生成方式を確定

### Phase 2: メモリ統合（2-3週間）

**目標**: ロングホライズンタスクでベースライン比2倍以上改善

1. MEM短期メモリ: ViTに空間-時間分離アテンション追加（6層ごとに時間アテンション挿入）
2. MEM長期メモリ: VLMがサブタスク指示+圧縮メモリを生成
3. DeltaNet再帰状態のメモリ的挙動の分析（プロービング、介入実験、CKA類似度）
4. 学習データ: サブタスクアノテーション+メモリラベル生成パイプライン

### Phase 3: 軽量RL改善パイプライン（2-3週間）

**目標**: RL適用後に成功率5-10%向上

**Low Policy: RLT**
- RLトークン抽出 → 小規模Actor-Criticで精密編集
- ManiSkillで予備検証 → 精密タスク設計（ネジ締め、コネクタ挿入の模擬）
- オンラインRL、15分〜2時間で収束目標
- フォールバック: Residual RL（状態ベースMLP残差ポリシー）

**High Policy: AWR or DPO**
- LRM（VLMベース報酬モデル）でサブタスク成功を判定
- アドバンテージ推定 → 重み付きSFTで再学習
- 推論コスト追加ゼロ

### Phase 4: 差別化要素の追加（2-3週間）

Phase 1-3の結果に応じて、以下から1-2個を選択:
- [ ] 離散拡散アクション生成（DIVA方式）
- [ ] ワールドモデル共進化（VLAW方式）
- [ ] 3D表現追加（SpatialVLA Ego3D PE）
- [ ] テスト時再帰精製（RD-VLA方式）

### Phase 5: 統合評価・論文（2-3週間）

アブレーションマトリクス:
| 構成 | メモリ | RL (Low: RLT) | RL (High: AWR) | 追加要素 | 評価先 |
|------|--------|---------------|----------------|---------|--------|
| Baseline | - | - | - | - | LIBERO |
| +MEM | ✓ | - | - | - | LIBERO-Long |
| +RLT | - | ✓ | - | - | 精密タスク |
| +AWR | - | - | ✓ | - | LIBERO |
| +MEM+RLT+AWR | ✓ | ✓ | ✓ | - | 全ベンチマーク |
| Full | ✓ | ✓ | ✓ | ✓ | 全ベンチマーク |

**総期間: 約3-4ヶ月**

---

## 8. 計算資源の見積もり

| フェーズ | GPU | 期間 | 備考 |
|---------|-----|------|------|
| Phase 1 SFT | 4-8×A100 | 1-2日 | LoRA、BF16 |
| Phase 2 MEM学習 | 8×A100 | 2-3日 | ビデオ-言語タスク含む |
| Phase 3 RLT | 1×オンボードGPU | 15分-2時間 | オンラインRL、軽量 |
| Phase 3 AWR | 4×A100 | 0.5-1日 | オフライン重み付きSFT |
| 評価 | 1×RTX 4090 | 随時 | 推論のみ |

---

## 9. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| DeltaNetのVLA FT不安定 | 高 | Full比率を上げた構成も検証。float32でSSM計算 |
| LIBERO飽和で差が出ない | 中 | CALVIN、実ロボット、カスタムタスクで差別化 |
| MEM再現困難 | 中 | MEMの論文記述+HAMLET等の代替メモリも検討 |
| RLT トークン抽出学習困難 | 中 | Residual RL（状態ベースMLP残差）にフォールバック |
| 推論速度未達 | 低 | Action Chunking増加、INT4量子化 |

---

## 参考: チーム議論での主要決定事項

1. **VLM**: Qwen3.5-4Bに確定。Qwen3-VLは比較対象
2. **ViT**: まずQwen native ViTで進める。C-RADIOは公式非推奨のため見送り
3. **メモリ**: MEM方式（短期ビデオ+長期言語）をベースに、DeltaNet再帰状態との相補性を探る
4. **RL**: RLT（Low Policy） + AWR/DPO（High Policy）の軽量構成。RECAPのCFGは速度面で不採用
5. **差別化**: 認知発達的カリキュラムpretrain + DeltaNet分析 + ゲーティング学習
6. **論文ターゲット**: CoRL 2026 / RSS 2026
