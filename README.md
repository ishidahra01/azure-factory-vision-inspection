# Azure Factory Vision Inspection

製造業向けのAI画像検査システム - Azure OpenAI GPT-4oを活用した部品外観検査の研究・実験プロジェクト

## 📋 プロジェクト概要

このプロジェクトは、製造業における部品の外観検査をAI（Azure OpenAI GPT-4o）で自動化するための実験を行っています。特に以下の4つのカテゴリでの分類精度向上を目指しています：

- **OK**: 正常な部品
- **汚れ**: 汚れが付着した部品
- **欠け**: 欠損がある部品  
- **削り節**: 削り節状の不良がある部品

## 🎯 技術的アプローチ

### 主要な実験手法

1. **段階的分類（Hierarchical Classification）**
   - Step 1: OK/NG の二値分類
   - Step 2: NG の場合、汚れ/加工不良 の二値分類
   - Step 3: 加工不良の場合、欠け/削り節 の二値分類

2. **Few-shot Learning**
   - サンプル画像を参考例として提供
   - GPT-4oの視覚的推論能力を活用

3. **画像前処理**
   - ガンマ補正、線形スケーリング
   - アンシャープマスク（エッジ強調）
   - 背景除去の効果検証

4. **モデル比較**
   - GPT-4o vs GPT-4o-mini
   - 精度・コスト・処理速度の比較

## 📁 プロジェクト構造

```text
azure-factory-vision-inspection/
├── 00_preprocessing.ipynb                              # 画像前処理実験
├── 01_vision_inspection_test_no_cut.ipynb             # 基本分類（背景除去なし）
├── 02_vision_inspection_test_no_cut_o4mini.ipynb      # reasoning model版
├── 03_vision_inspection_test_no_cut_binary_fewshot.ipynb           # 段階分類
├── 03_vision_inspection_test_no_cut_binary_fewshot_collage.ipynb   # コラージュ版
├── 03_vision_inspection_test_no_cut_binary_fewshot_cv.ipynb        # 画像前処理統合版
├── 04_vision_inspection_test_no_cut_o4mini_binary_fewshot.ipynb    # reasoning model版段階分類
├── 05_vision_inspection_test_cut_binary_fewshot.ipynb              # 背景除去版
├── requirements.txt                                    # 依存パッケージ
├── analysis-results/                                   # 実験結果
├── input/                                             # 入力画像データセット（事前配置必要）
│   ├── 変換後/                                        # 前処理済み画像
│   ├── 背景カットあり/                                  # 背景除去済み画像
│   └── 背景カットなし/                                  # 元画像
└── temp/                                              # 実験用一時ファイル
```

## 📚 各ノートブックの詳細説明

### 00_preprocessing.ipynb

**目的**: 画像前処理手法の検証  
**内容**:

- ガンマ補正、明度・コントラスト調整
- ヒストグラム均等化、CLAHE（適応的ヒストグラム均等化）
- ノイズ除去、アンシャープマスク
- クロップ・リサイズ処理
- 各前処理の効果を視覚的に比較

### 01_vision_inspection_test_no_cut.ipynb

**目的**: 基本的な4クラス分類の実装と評価  
**特徴**:

- GPT-4.1による直接的な4クラス分類（OK/汚れ/欠け/削り節）
- Azure Blob Storageを活用した画像管理
- 背景除去なしの元画像を使用
- ベースライン性能の確立

### 02_vision_inspection_test_no_cut_o4mini.ipynb

**目的**: reasoning model(o4-mini)の性能評価  
**特徴**:

- GPT-4.1とo4-miniの比較実験
- コスト効率と精度のトレードオフ分析
- 01番と同じ4クラス直接分類方式

### 03_vision_inspection_test_no_cut_binary_fewshot.ipynb

**目的**: 段階的二値分類アプローチの実装  
**特徴**:

- 3段階の二値分類による精度向上
- Few-shotサンプルを用いた推論精度向上
- Pydanticモデルによる構造化された出力
- 各段階の信頼度スコア付き

### 03_vision_inspection_test_no_cut_binary_fewshot_collage.ipynb

**目的**: 比較画像コラージュによる精度向上  
**特徴**:

- サンプル画像と対象画像を並べたコラージュ作成
- 視覚的比較による判断精度の向上
- PIL/Pillowを使ったコラージュ生成機能
- より直感的な画像比較手法

### 03_vision_inspection_test_no_cut_binary_fewshot_cv.ipynb

**目的**: 画像前処理と分類の統合実験  
**特徴**:

- OpenCVベースの前処理パイプライン
- 日本語パス対応の画像読み込み
- 前処理パラメータの最適化
- 処理済み画像での分類性能評価

### 04_vision_inspection_test_no_cut_o4mini_binary_fewshot.ipynb

**目的**: o4-miniでの段階分類性能評価  
**特徴**:

- o4-miniでの段階的二値分類
- コスト効率の良い高精度分類の実現
- GPT-4.1との性能比較分析

### 05_vision_inspection_test_cut_binary_fewshot.ipynb

**目的**: 背景除去の効果検証  
**特徴**:

- 背景除去済み画像での分類実験
- 前処理効果の定量的評価
- 背景ノイズの影響分析

## 🛠️ 技術スタック

- **AI**: Azure OpenAI (GPT-4.1, o4-mini)
- **画像処理**: OpenCV, PIL/Pillow
- **データ分析**: Pandas, NumPy, Scikit-learn
- **可視化**: Matplotlib, Seaborn
- **クラウド**: Azure Blob Storage
- **構造化出力**: Pydantic
- **並行処理**: ThreadPoolExecutor

## 🚀 セットアップ手順

### 1. 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/ishidahra01/azure-factory-vision-inspection.git
cd azure-factory-vision-inspection

# 仮想環境の作成（推奨）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env` ファイルを作成し、以下を設定：

```env
# Azure OpenAI設定
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key

# Azure Blob Storage設定
BLOB_CONNECTION_STRING=your-connection-string
BLOB_CONTAINER_NAME=your-container-name
```

### 3. データセットの準備

画像を以下の構造で配置：

```text
input/
├── 背景カットなし/
│   ├── sample/    # Few-shot用サンプル画像
│   │   ├── OK/
│   │   ├── 削り節/
│   │   ├── 欠け/
│   │   └── 汚れ/
│   └── test/      # テスト用画像
│       ├── OK/
│       ├── 削り節/
│       ├── 欠け/
│       └── 汚れ/
└── 背景カットあり/
    └── (同様の構造)
```

## 📊 実験結果

実験結果は `analysis-results/` フォルダに保存されます：

- **classification_results.csv**: 基本分類結果
- **hierarchical_classification_results_[timestamp].csv**: 段階分類結果
- 各ファイルには予測精度、混同行列、信頼度スコアなどが含まれます

## 🎯 主要な発見・知見

1. **段階的分類の効果**: 直接4クラス分類よりも段階的二値分類の方が高精度
2. **Few-shotの効果**: 参考画像の提供により分類精度が大幅向上
3. **前処理の重要性**: 適切な画像前処理により認識精度が改善
4. **モデル比較**: GPT-4o vs GPT-4o-miniのコスト・精度トレードオフ
5. **背景除去の効果**: 対象物への集中により分類精度向上