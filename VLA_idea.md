# VLA構想

## 構想案（原案）

Stage 1 High Policy
inputs:
  - short video clip
  - full task goal
  - previous long-term language memory

backbone:
  - video-capable VLM
  - short-term temporal info is compressed into current-step visual representation

outputs:
  - current subtask
  - updated long-term language memory

## 原案に対する議論

### 問題点1: 「short video clip」の定義が曖昧

5Hzで動作するHigh Policyに「short video clip」を入力するとして、以下が未定義:
- **フレーム数**: 何フレームか（MEMでは6フレーム事前学習 → 18フレーム拡張）
- **取得間隔**: 1Hz × 6フレーム = 6秒の履歴 vs 5Hz × 6フレーム = 1.2秒の履歴。システムの性質が根本的に変わる
- **フレーム選択戦略**: 等間隔か、イベントドリブンか、適応的か

→ **解決策**: 取得間隔は可変とし、pretrain段階で段階的に拡張する設計を採用。初期は1Hz × 6フレーム（6秒窓）から開始し、後期に5Hz × 18フレーム（3.6秒窓・高解像度）まで拡張。

### 問題点2: 長期言語メモリの更新条件が未定義

「previous long-term language memory → updated long-term language memory」とあるが:
- **毎ステップ更新** vs **サブタスク完了時のみ更新** が未定義
- MEMの「成功時のみ更新」はヒューリスティックであり、最適な更新タイミングは学習可能であるべき

→ **解決策**: ワーキングメモリゲーティング（03_research_proposal.md アイデア2参照）を導入。基底核的なゲーティングネットワークで「いつ更新するか」を学習。初期はMEMのヒューリスティックで始め、Phase 1-Cで学習可能なゲーティングに置換。

### 問題点3: 「VLMの言語化能力をできるだけ引き継ぐ」の具体化不足

ロボットデータでpretrainした瞬間にVLMの汎用言語能力は劣化する。防止策が必要:
- **LoRA分離**: ロボットドメイン専用のLoRAアダプタで汎用重みを保護
- **Rehearsal**: pretrainデータの10-20%をWebテキスト・画像データで混合
- **段階的言語導入**: 初期は言語条件を外して視覚-行動模倣のみ、後期に言語条件付けを導入（認知発達的アプローチ）

→ **解決策**: Phase 1-Aで言語ヘッドを凍結し視覚-運動接地のみ実施。Phase 1-B以降で段階的に言語を導入。

### 問題点4: pretrainとfinetuneの境界が不明確

Stage 1は構造定義であり「どのデータでどの順序で学習するか」（pretrain戦略）が議論されていない。

→ **解決策**: 以下の修正案で3サブフェーズに分割。

---

## Stage 1 修正案: 認知発達的カリキュラムpretrain

### 設計原理

Piagetの感覚運動段階理論 [Piaget, 1952] とElmanの「小さく始める」原則 [Elman, 1993] に基づき、pretrainを3サブフェーズに分割する。各フェーズは前フェーズの能力を前提とし、段階的に複雑な能力を獲得する。

```
Phase 1-A ──► Phase 1-B ──► Phase 1-C ──► [Stage 2: アクション生成]
視覚-運動接地   サブタスク分解   メモリ管理精緻化     後段で選定
(見て理解)     (計画を立てる)  (記憶を管理する)
```

### Phase 1-A: 視覚-運動接地 (Visual-Motor Grounding)

```yaml
目標: ViT + LLMの視覚-運動対応を確立（行動出力なし）
データ: OXE/DROID の単純操作 (pick, place, push)
フレーム: 6 frames @ 1Hz（6秒窓）

学習構成:
  ViT: temporal attention追加、学習可能
  Projector: 学習可能
  LLM (DeltaNet+Full): LoRAのみ（低学習率）
  言語ヘッド: 凍結（VLM言語能力を保護）
  アクションヘッド: なし（この段階では未導入）

補助損失:
  - 次フレーム予測（予測的処理の獲得、アイデア9に対応）
  - アフォーダンス予測（「掴めるか」「置けるか」、アイデア7に対応）
  - 遮蔽物体追跡（対象永続性、アイデア9に対応）

入力: 6 frames @ 1Hz + task description (短い, "pick up the cup")
出力: 行動記述テキスト ("reaching toward red cup on left")

検証ポイント:
  - DeltaNet再帰状態が遮蔽中の物体情報を保持するか
  - ViT temporal attentionの効果（有/無のアブレーション）
  - 言語ヘッド凍結による汎用言語能力の維持度
```

**認知発達的対応**: 乳児の一次循環反応期（生後1-4ヶ月）に対応。自己の行動と外界の関係を学ぶ段階。言語は使わず、視覚と運動の対応のみを獲得する。

### Phase 1-B: サブタスク分解の獲得

```yaml
目標: 長期タスクのサブタスク分解能力を獲得
データ: サブタスクアノテーション付きデモ（LLMで自動生成）
フレーム: 12 frames @ 1Hz（12秒窓、段階的拡張）

学習構成:
  ViT: Phase 1-Aから継続学習
  Projector: 継続学習
  LLM (DeltaNet+Full): LoRA学習率を上げる
  言語ヘッド: 凍結解除（段階的にunfreeze）
  アクションヘッド: なし（まだ未導入）

新規タスク:
  - サブタスク分解（長期タスクを段階に分割）
  - メモリ生成（「これまでに何をしたか」の要約）
  - 意図推定（同じゴールの異なる達成手段を理解）

入力: 12 frames @ 1Hz + full task goal + (空の)memory
出力: current subtask + updated memory

検証ポイント:
  - サブタスク分解の精度（人手アノテーションとの一致度）
  - メモリ生成の情報量と圧縮率
  - DeltaNet再帰状態の変化パターン（サブタスク境界で変化するか）
```

**認知発達的対応**: 二次循環反応期～手段-目的関係期（生後8-18ヶ月）に対応。目的達成のための手順を理解し始める段階。言語理解が始まるが、まだ言語で行動を指示されるには至らない。

### Phase 1-C: メモリ管理の精緻化

```yaml
目標: 適応的メモリ更新と確信度推定を学習
データ: Phase 1-Bの出力 + 失敗ケースを含む多様なエピソード
フレーム: 18 frames @ variable Hz（可変取得間隔）

学習構成:
  ViT: 継続学習（フレーム数拡張に対応）
  Projector: 継続学習
  LLM: Full fine-tuning（LoRAから移行）
  言語ヘッド: 完全解放
  ゲーティングネットワーク: 新規導入（メモリ更新タイミングの学習）

新規要素:
  - ゲーティングネットワーク（アイデア2: PFC-BGゲーティング）
  - 確信度スコア出力（FEP精度重み付けの簡易版、アイデア5）
  - Rehearsal: 10-20%のWebデータ混合（VLM言語能力維持）

入力: 18 frames @ variable Hz + task goal + memory_{t-1}
出力: subtask + memory_t + confidence score

検証ポイント:
  - ゲーティングネットワーク vs ヒューリスティック（MEMの「成功時のみ」）の比較
  - 確信度スコアとタスク成功率の相関
  - Rehearsal比率とVLM言語能力のトレードオフ
  - 18フレームへの拡張時の性能変化
```

**認知発達的対応**: 表象的思考期（18ヶ月以降）に対応。記憶の管理、計画の修正、失敗からの学習が可能になる段階。言語指示の理解と実行が可能。

### Phase 1 → Stage 2 判断ポイント

Phase 1-C完了後、以下を基にアクション生成方式を選定:
- DeltaNetの振る舞い分析結果 → 離散トークン vs 連続値の適性判断
- 推論速度測定 → 30Hz/100Hz達成可能性の確認
- MTP（Multi-Token Prediction）の有効性 → 離散拡散との相性確認

```
選択肢:
├─ Flow Matching (DiT) — PIと同系統だがDeltaNet上での挙動が差別化要素
├─ 離散拡散 (DIVA/DD-VLA方式) — MTPとの親和性が高い場合に有利
├─ ActionCodec (RVQトークナイザ) — 最もシンプルで推論速度が出やすい
└─ Hopfield連想記憶 — エネルギーベースで根本的に異なるアプローチ（アイデア4）
```

---

## 目指すべきもの
- 成功率が高い
- 研究要素がありSOTAが狙えて社会実装が見込めるもの
- VLMの言語化能力はできるだけ引き継ぐ
  → **具体策**: Phase 1-Aで言語ヘッド凍結、段階的解放、Rehearsal混合
- ロングタスクができる
  → **具体策**: Phase 1-B/Cでサブタスク分解+適応的メモリ管理を獲得
- ツールの使用ができる
  → **具体策**: Phase 1-Bの「手段-目的関係」段階で道具使用タスクを含める
- 強化学習を併用した成功率改善パイプラインがある
  → **具体策**: Low Policy: RLT（<1ms、オンラインRL 15分〜2時間で収束）。High Policy: AWR/DPO（推論コスト追加ゼロ、RECAPのCFGは速度面で不採用）。詳細は03_research_proposal.mdセクション5参照
- 高ポリシーは5hz程度　下位ポリシーは30hz以上で回したい，理想は100hz
  → **具体策**: DeltaNetの線形アテンションで5Hz High Policy実現。Low PolicyはStage 2で DiT FM + RLT精密編集（RLTは<1msでボトルネックにならない）

## 参考
- qwen3.5 4B (利用しようとしてるVLM)
- https://huggingface.co/nvidia/C-RADIOv4-SO400M 使えたら面白そうなViT
- RLT https://www.pi.website/download/rlt.pdf
- mem https://www.pi.website/download/Mem.pdf
- xiaomi robotics-0
- pi0.5，pi0.6

## 認知発達・認知科学の主要参考文献（pretrain設計の理論的根拠）
- Piaget, J. (1952). *The Origins of Intelligence in Children*. — 感覚運動段階理論
- Elman, J.L. (1993). "Learning and development in neural networks: the importance of starting small." *Cognition*, 48(1), 71-99. — 段階的学習の計算的優位性
- Gibson, J.J. (1979). *The Ecological Approach to Visual Perception*. — アフォーダンス理論
- O'Reilly, R.C. & Frank, M.J. (2006). "Making Working Memory Work." *Neural Computation*, 18(2), 283-328. — PFC-BGゲーティング
- Baillargeon, R. (1987). "Object permanence in 3½- and 4½-month-old infants." *Developmental Psychology*, 23(5), 655-664. — 対象永続性
- Meltzoff, A.N. (1988). "Infant imitation after a 1-week delay." *Developmental Psychology*, 24(4), 470-476. — 模倣の段階性
- Tomasello, M. (1999). *The Cultural Origins of Human Cognition*. — 社会的学習の階層
