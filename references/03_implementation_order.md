# 実装順序提案書【アーカイブ】

> **注意**: 本文書の内容は `docs/03_research_proposal.md` に統合済み。提案書のセクション7（実装計画）が最新版。
> 本ファイルは参照用にのみ保持。新たな編集は提案書側で行うこと。

**作成日**: 2026年3月29日

---

## 実装フェーズ全体像

```
Phase 0 (1週間)     Phase 1 (2-3週間)     Phase 2 (2-3週間)
┌──────────┐      ┌──────────────┐      ┌──────────────┐
│ 環境構築  │─────►│ ベースVLA    │─────►│ MEM統合      │
│ データ準備│      │ (Qwen+FM)   │      │ (短期+長期)  │
└──────────┘      └──────────────┘      └──────────────┘
                                               │
Phase 5 (2-3週間)   Phase 4 (2-3週間)    Phase 3 (2-3週間)
┌──────────────┐  ┌──────────────┐      ┌──────────────┐
│ RL統合       │◄─│ RLT統合      │◄─────│ C-RADIO置換  │
│ (RECAP)      │  │ (低ポリシー) │      │ (ViT swap)   │
└──────┬───────┘  └──────────────┘      └──────────────┘
       │
       ▼
Phase 6 (2-3週間)
┌──────────────┐
│ 統合評価     │
│ 論文執筆     │
└──────────────┘

総期間: 約3-5ヶ月
```

---

## Phase 0: 環境構築・データ準備（1週間）

### 0.1 開発環境
```bash
# Python環境
uv venv .venv && source .venv/bin/activate
uv pip install torch torchvision transformers accelerate
uv pip install lerobot simpler-env mani-skill  # シミュレータ
uv pip install wandb hydra-core  # 実験管理
```

### 0.2 データセット準備
| データセット | 用途 | サイズ |
|-------------|------|--------|
| LIBERO | 評価ベンチマーク | 4タスクスイート×50デモ |
| DROID | ロボットSFT事前学習 | 76k デモ |
| Bridge V2 | 多様な操作データ | 60k デモ |
| OXE (Open X-Embodiment) | 大規模事前学習 | 800k+ エピソード |

### 0.3 モデル準備
```bash
# Qwen3.5-4B (メインターゲット — 線形アテンション搭載、DeepStack廃止)
huggingface-cli download Qwen/Qwen3.5-4B-Instruct

# Qwen3-VL-4B (比較対象 — DeepStack有、フルアテンション)
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct

# C-RADIO (Phase 3用)
huggingface-cli download nvidia/C-RADIOv4-SO400M
```

### 0.4 評価環境
- LIBERO (MuJoCo)
- SimplerEnv (Google RT環境の再現)
- CALVIN (長期タスク)

**マイルストーン**: 全環境で既存モデル（OpenVLA-OFT等）の再現実験が完了

---

## Phase 1: ベースVLA構築（2-3週間）

### 1.1 目標
**Qwen3.5-4B** + フローマッチングアクションヘッドの基本VLAを構築し、ベースライン性能を確立する。Qwen3.5はDeepStackが廃止されておりViT出力がクリーン、Gated DeltaNet（線形アテンション）による高速推論が可能。

### 1.2 実装タスク

#### Task 1.1: アクションエキスパート実装
- DiT (Diffusion Transformer) 16層
- 入力: VLM最終層KVキャッシュ + 固有受容覚 + ノイズ付き行動
- 出力: アクションチャンク（チャンクサイズ=10、50Hzなら0.2秒分）
- フローマッチング損失で学習

#### Task 1.2: 学習パイプライン（Stage 1: Alignment）
- Qwen ViT (frozen) → Projector (train) → Qwen LLM (frozen)
- ロボットデータでProjectorのアライメント
- 行動トークンをFASTトークン化で離散表現

#### Task 1.3: 学習パイプライン（Stage 2: SFT）
- 全体を微調整（LoRA適用）
- テレオペレーションデモデータでSFT
- 非同期実行技術（Xiaomi-R0方式）の導入

#### Task 1.4: 評価
- LIBEROで4スイート評価
- 目標: 90%+ (OpenVLA-OFT 97.1%に近づく)

**マイルストーン**: LIBERO 90%+ 達成

---

## Phase 2: MEM統合（2-3週間）

### 2.1 目標
短期ビデオメモリと長期言語メモリをPhase 1のベースVLAに統合する。

### 2.2 実装タスク

#### Task 2.1: 短期ビデオメモリ（ViT改修）
```python
# 概要: ViTに空間-時間分離アテンションを追加
# 4層ごとに因果的時間アテンションを挿入
# 追加パラメータなし（正弦波時間位置エンコーディング）

class SpatioTemporalViT(nn.Module):
    """
    Qwen ViTを改修:
    - 標準ViTレイヤは空間アテンション（全タイムステップのパッチ）
    - 4層ごとに因果的時間アテンション挿入
    - 上層で過去フレームのパッチをドロップ
    """
```

- 正弦波時間位置エンコーディング（t=0で値0、単一画像と等価）
- 6フレーム（事前学習）→ 18フレーム（ポストトレーニング拡張）
- 計算量: O(Kn² + nK²)

#### Task 2.2: 長期言語メモリ
```python
# 高ポリシーの出力を拡張:
# (subtask) → (subtask, updated_memory)
# メモリはVLMの自然言語テキストとして表現

# 学習ターゲット生成パイプライン:
# 1. エピソードをサブタスクにアノテーション
# 2. LLMで将来のタスク実行に関連する情報を要約
# 3. 圧縮（冗長な詳細を削除）
```

- 成功時のみメモリ更新（失敗の繰り返しで未観測シーケンス生成を防止）
- メモリ圧縮: LLMによる情報の削除・圧縮

#### Task 2.3: 階層的分解
- 高ポリシー (π_HL): サブタスク指示 + メモリ更新を生成（5Hz）
- 低ポリシー (π_LL): サブタスク指示に条件付けてアクション生成（30Hz+）
- 非同期実行: 高ポリシーは低頻度、低ポリシーは高頻度で独立実行

#### Task 2.4: 学習
- 事前学習段階でビデオ-言語タスクを含める（MEMの知見：事前学習なしでは45%に低下）
- メモリタスク用のデータ生成（サブタスクアノテーション + メモリラベル）

#### Task 2.5: 評価
- LIBERO-Long（長期タスク）
- メモリ能力テスト（カウント、探索、状態追跡）
- 短期タスクでの性能劣化がないことを確認

**マイルストーン**: ロングホライズンタスクでベースライン比2倍以上の改善

---

## Phase 3: C-RADIO ViT置換（2-3週間）

### 3.1 目標
Qwen VLMの内蔵ViTをC-RADIOv4-SO400Mに置換し、性能比較を行う。

### 3.2 実装タスク

#### Task 3.1: Projector再設計
```python
class CRadioProjector(nn.Module):
    """
    C-RADIO (dim=1152) → Qwen LLM (dim=2048 or 2560)
    """
    def __init__(self, radio_dim=1152, llm_dim=2560):
        self.spatial_merge = nn.Conv2d(radio_dim, radio_dim, 2, stride=2)  # 2x2 merge
        self.proj = nn.Sequential(
            nn.Linear(radio_dim * 4, llm_dim),  # 4 = merge後の結合
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
```

#### Task 3.2: 動画処理の追加
- C-RADIOは静止画モデル → フレームごとに適用
- Phase 2の時間アテンションをC-RADIO出力に適用
- temporal_patch_size相当の処理をProjectorに統合

#### Task 3.3: Projector設計（Qwen3.5用 — DeepStack不要）
- Qwen3.5ではDeepStackが廃止されているため、シンプルな設計で済む
- C-RADIO (dim=1152) → 2層MLP → Qwen3.5 LLM (dim=2560)
- Spatial Merge (2x2) をProjectorに統合
- ※Qwen3-VL比較実験時のみDeepStack代替を検討

#### Task 3.4: 段階的学習
1. **Stage 1**: C-RADIO (frozen) + Projector (train) + LLM (frozen)
2. **Stage 2**: C-RADIO (frozen) + Projector (train) + LLM (train, LoRA)
3. **Stage 3**: C-RADIO (very low-lr) + Projector (train) + LLM (train)

#### Task 3.5: 比較実験
- Qwen native ViT vs C-RADIO（同一学習条件）
- 空間特徴エントロピーの測定
- タスク別性能分析（空間精度が重要なタスクでの差異に注目）

**マイルストーン**: C-RADIO版がQwen ViT版と同等以上の性能

---

## Phase 4: RLT統合（2-3週間）

### 4.1 目標
低ポリシーの精密操作タスクをRLTで改善する。

### 4.2 実装タスク

#### Task 4.1: RLトークン抽出器
```python
class RLTokenExtractor(nn.Module):
    """
    VLA内部埋め込みからRLトークンを抽出する
    エンコーダ-デコーダTransformer（情報ボトルネック）
    """
    def __init__(self, vla_dim, rl_token_dim, num_tokens):
        self.encoder = TransformerEncoder(vla_dim, rl_token_dim, num_tokens)
        self.decoder = TransformerDecoder(rl_token_dim, vla_dim)

    # 学習: 再構成損失で事前学習
```

#### Task 4.2: Actor-Critic
```python
class RLTActorCritic(nn.Module):
    """
    RLトークン上で動作する小規模Actor-Critic
    Actorはアクションの「編集」を出力
    """
    def __init__(self, rl_token_dim, action_dim):
        self.actor = MLP(rl_token_dim + action_dim, action_dim)  # VLA予測+RLトークン→編集
        self.critic = MLP(rl_token_dim, 1)  # 状態価値
```

- Reference-action dropout（学習初期の単純コピー防止）
- オフポリシーRL（毎秒数百回更新）
- VLA予測に基づく探索（Criticが改善を検出した場合のみ逸脱）

#### Task 4.3: 実機/シミュレーション統合
- シミュレーション（ManiSkill）での予備検証
- 精密タスク設計（ネジ締め、コネクタ挿入の模擬環境）

#### Task 4.4: 評価
- ベースライン比の改善倍率
- 学習時間（目標: 15分〜2時間で収束）
- 100Hz制御の達成確認

**マイルストーン**: 精密タスクでベースライン比3倍以上の改善

---

## Phase 5: RECAP RL統合（2-3週間）

### 5.1 目標
高ポリシーにRECAP的アドバンテージ条件付けを導入し、全体性能を改善する。

### 5.2 実装タスク

#### Task 5.1: アドバンテージ条件付け
- VLMの入力テキストに「Advantage: positive/negative」を追加
- 全データ（成功+失敗）から学習し、推論時にpositive方向にガイド

#### Task 5.2: 価値関数
- 小規模VLM（Qwen-0.5B相当 or MLP）で価値関数を構築
- 201ビン離散化分布的価値関数
- タスク完了までのステップ数を予測

#### Task 5.3: 分類器フリーガイダンス
```python
# 推論時:
# π*(a|o,l) ∝ π_ref(a|o,l) × (π_ref(a|I,o,l) / π_ref(a|o,l))^β
# β: ガイダンス強度（ハイパーパラメータ）
```

#### Task 5.4: 自律データ収集ループ
1. ポリシーで自律走行
2. 専門家介入（テレオペによる修正）を記録
3. 価値関数を更新
4. アドバンテージ条件付きでポリシー更新
5. 繰り返し

**マイルストーン**: RECAP適用後、全体成功率5-10%向上

---

## Phase 6: 統合評価・論文執筆（2-3週間）

### 6.1 統合実験
- 全コンポーネント統合後のシステム評価
- アブレーションスタディ（各コンポーネントの寄与度）
- ベンチマーク: LIBERO, LIBERO-Long, SimplerEnv, CALVIN

### 6.2 アブレーション実験マトリクス
| 構成 | C-RADIO | MEM | RLT | RECAP | 評価 |
|------|---------|-----|-----|-------|------|
| Baseline | - | - | - | - | LIBERO |
| +C-RADIO | ✓ | - | - | - | LIBERO |
| +MEM | - | ✓ | - | - | LIBERO-Long |
| +C-RADIO+MEM | ✓ | ✓ | - | - | LIBERO-Long |
| +RLT | - | - | ✓ | - | 精密タスク |
| +RECAP | - | - | - | ✓ | 全体 |
| Full | ✓ | ✓ | ✓ | ✓ | 全ベンチマーク |

### 6.3 論文構成
1. Introduction: 統合階層VLAの必要性
2. Related Work: VLA動向、メモリ機構、RL for VLA
3. Method: C-RADIO ViT置換、MEM統合、RLT/RECAP
4. Experiments: ベンチマーク評価、アブレーション
5. Discussion: 空間特徴エントロピーの分析
6. Conclusion

**マイルストーン**: 論文ドラフト完成、実験結果の取りまとめ

---

## リスク管理

| リスク | 影響度 | 対策 |
|--------|--------|------|
| C-RADIO置換で性能低下 | 高 | Qwen ViTをフォールバックとして維持 |
| MEM学習データ不足 | 中 | LLMによるサブタスクアノテーション自動生成 |
| RLTの報酬設計困難 | 中 | 再構成損失ベースの半教師あり学習 |
| 推論速度未達 | 低 | INT4量子化、Action Chunking増加 |
| LIBERO飽和 | 中 | CALVIN、実ロボット、カスタムタスクで差別化 |
